import unittest

from poker_rl.table_session import TableSession


class Learner:
    def __init__(self):
        self.transitions = []
        self.flags = []
        self.starts = []
        self.ends = []
        self.updates = 0

    def act(self, observation, action_mask, *, training=False):
        self.flags.append(training)
        return 1 if action_mask[1] else 2

    def observe(self, transition):
        self.transitions.append(transition)
        self.updates += 1

    def on_episode_start(self, *, seed, training):
        self.starts.append(training)

    def on_episode_end(self, *, reward, terminated, truncated, training):
        self.ends.append((reward, training))


class NonLearner:
    def act(self, observation, action_mask, *, training=False):
        return 1 if action_mask[1] else 2


class Registry:
    def load(self, name, kwargs):
        return Learner() if name == "learner" else NonLearner()


class TableTrainingTests(unittest.TestCase):
    def session(self, count=2, **kwargs):
        session = TableSession(Registry(), [("learner", {})] * count, **kwargs)
        self.addCleanup(session.close)
        return session

    def finish(self, session):
        while not session.game.finished:
            session.play(session.choose_action())
        session.finish_hand()

    def test_only_selected_seats_learn_at_all_table_sizes(self):
        for count in range(2, 6):
            session = self.session(count, mode="train", training_seats=(0, count - 1))
            for hand in range(3):
                if hand:
                    session.next_hand()
                self.finish(session)
            self.assertEqual(session.stats.completed, 3)
            for seat, agent in enumerate(session.agents):
                selected = seat in (0, count - 1)
                self.assertEqual(agent.starts, [selected] * 3)
                self.assertEqual([flag for _, flag in agent.ends], [selected] * 3)
                self.assertTrue(all(flag == selected for flag in agent.flags))
                self.assertEqual(session.stats.transitions[seat], len(agent.transitions))
                self.assertEqual(len([t for t in agent.transitions if t.terminated]), 3 if selected else 0)
                for transition in agent.transitions:
                    self.assertEqual(transition.info["seat"], seat)
                    self.assertTrue(transition.action_mask[transition.action])
                    if not transition.terminated:
                        self.assertEqual(transition.info["current_player"], seat)
                        self.assertEqual(transition.reward, 0)
                    else:
                        self.assertFalse(transition.next_action_mask.any())
            self.assertAlmostEqual(sum(session.stats.net_bb), 0)

    def test_inference_never_updates_even_when_training_seats_are_supplied(self):
        session = self.session(5, mode="infer", training_seats=(0, 1))
        self.finish(session)
        for agent in session.agents:
            self.assertEqual(agent.updates, 0)
            self.assertFalse(any(agent.flags + agent.starts))

    def test_watching_keeps_learned_instances_and_stops_learning(self):
        session = self.session(3, mode="train", training_seats=(0, 2))
        session.run_batch(5)
        self.assertEqual(session.stats.completed, 5)
        identities = [id(agent) for agent in session.agents]
        updates = [agent.updates for agent in session.agents]
        self.assertGreater(updates[0], 0)
        session.use_inference()
        self.assertEqual(session.mode, "infer")
        self.assertEqual([id(agent) for agent in session.agents], identities)
        self.finish(session)
        self.assertEqual([agent.updates for agent in session.agents], updates)
        self.assertTrue(all(agent.starts[-1] is False for agent in session.agents))
        self.assertTrue(all(agent.ends[-1][1] is False for agent in session.agents))

    def test_fast_batches_match_visual_training_and_stop_exactly_at_target(self):
        visual = self.session(5, mode="train", training_seats=(0, 4))
        fast = self.session(5, mode="train", training_seats=(0, 4), defer_callbacks=True)
        for hand in range(7):
            if hand:
                visual.next_hand()
            self.finish(visual)
        while fast.stats.hands < 7:
            fast.run_batch(7, max_actions=13)
        self.assertEqual(visual.stats, fast.stats)
        counts = fast.stats
        fast.run_batch(7)
        fast.finish_hand()
        self.assertEqual(fast.stats, counts)

    def test_finish_callbacks_can_be_deferred_without_losing_final_reward(self):
        session = self.session(mode="train", training_seats=(0, 1), defer_callbacks=True)
        while not session.game.finished:
            session.play(session.choose_action())
        self.assertEqual(session.stats.completed, 0)
        self.assertFalse(any(t.terminated for a in session.agents for t in a.transitions))
        with self.assertRaises(RuntimeError):
            session.next_hand()
        session.finish_hand()
        self.assertEqual(session.stats.completed, 1)
        for seat, agent in enumerate(session.agents):
            self.assertEqual(agent.transitions[-1].reward, session.game.rewards()[seat])
            self.assertTrue(agent.transitions[-1].terminated)
        with self.assertRaises(RuntimeError):
            session.play(1)

    def test_pause_and_stop_do_not_advance_more_actions(self):
        session = self.session(mode="train", training_seats=(0,))
        session.pause_event.set()
        session.run_batch(10)
        self.assertEqual(session.game.action_count, 0)
        session.pause_event.clear()
        session.run_batch(10, max_actions=1)
        self.assertEqual(session.game.action_count, 1)
        session.stop_event.set()
        session.run_batch(10)
        self.assertEqual(session.game.action_count, 1)

    def test_truncated_hand_is_delivered_and_counted_separately(self):
        session = self.session(mode="train", training_seats=(0, 1), game_kwargs={"max_actions": 1})
        self.finish(session)
        self.assertEqual(session.stats.completed, 0)
        self.assertEqual(session.stats.truncated, 1)
        transitions = [t for agent in session.agents for t in agent.transitions]
        self.assertEqual(len(transitions), 1)
        self.assertTrue(transitions[0].truncated)
        self.assertEqual(transitions[0].reward, 0)

    def test_training_requires_valid_selected_agents(self):
        for seats in ((), (2,)):
            with self.assertRaises(ValueError):
                self.session(mode="train", training_seats=seats)
        with self.assertRaisesRegex(TypeError, "observe"):
            TableSession(Registry(), [("other", {}), ("learner", {})], mode="train", training_seats=(0,))


if __name__ == "__main__":
    unittest.main()
