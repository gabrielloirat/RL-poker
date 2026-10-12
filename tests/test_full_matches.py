import unittest

from poker_rl import AgentRegistry, PokerGame
from poker_rl.table_session import TableSession


class Learner:
    def __init__(self):
        self.transitions = []
        self.flags = []

    def act(self, observation, action_mask, *, training=False):
        self.flags.append(training)
        return 1 if action_mask[1] else 2

    def observe(self, transition):
        self.transitions.append(transition)


class Registry:
    def load(self, name, kwargs):
        return Learner()


class FullMatchTests(unittest.TestCase):
    def make_session(self, count=2, **kwargs):
        session = TableSession(AgentRegistry(), [("random", {"seed": seat + 7}) for seat in range(count)],
                               continuous=True, **kwargs)
        self.addCleanup(session.close)
        return session

    def finish(self, session):
        while not session.game.finished:
            session.play(session.choose_action())
        session.finish_hand()

    def test_one_hand_is_not_a_match_and_stacks_carry_forward(self):
        session = self.make_session()
        session.choose_action()
        session.play(0)
        previous = tuple(player.stack for player in session.game.table.players)
        self.assertEqual(previous, (99, 101))
        self.assertFalse(session.match_finished)
        self.assertEqual(session.stats.matches, 0)
        self.assertEqual(session.stats.hands, 1)
        session.advance()
        self.assertEqual(session.match, 1)
        self.assertEqual(session.hand_in_match, 2)
        self.assertEqual(session.game.hand_start_stacks, previous)
        self.assertEqual(session.game.table.dealer_pos, 1)
        self.assertEqual(tuple(player.stack for player in session.game.table.players), (97, 100))
        self.finish(session)
        for seat, player in enumerate(session.game.table.players):
            self.assertAlmostEqual(session.game.rewards()[seat], (player.stack - previous[seat]) / 2)
            self.assertAlmostEqual(session.stats.net_bb[seat], (player.stack - 100) / 2)

    def test_busted_seats_are_skipped_and_two_survivors_play_heads_up(self):
        game = PokerGame(num_players=5, initial_stacks=20)
        self.addCleanup(game.close)
        game.reset(seed=1, options={"dealer_seat": 0})
        while not game.finished:
            game.step(1 if game.action_masks()[1] else 2)

        for player, stack in zip(game.table.players, (0, 50, 0, 0, 50)):
            player.stack = stack
        game.start_next_hand(seed=2)
        self.assertEqual(game.table.dealer_pos, 1)
        self.assertEqual(game.current_seat, 1)
        self.assertEqual(game.table.player_pots, [0, 1, 0, 0, 2])
        for seat in (0, 2, 3):
            self.assertEqual(game.table.players[seat].cards, [])
            self.assertFalse(game.action_masks(seat).any())
        while not game.finished:
            self.assertIn(game.current_seat, (1, 4))
            game.step(1 if game.action_masks()[1] else 2)
        self.assertAlmostEqual(sum(player.stack for player in game.table.players), 100)

    def test_complete_matches_at_all_table_sizes_conserve_chips_and_count_one_winner(self):
        for count in range(2, 6):
            session = self.make_session(count, game_kwargs={"initial_stacks": 20})
            busted = set()
            for hand in range(1000):
                while not session.game.finished:
                    self.assertNotIn(session.game.current_seat, busted)
                    session.play(session.choose_action())
                    pot = 0 if session.game.terminated else session.game.table.community_pot + session.game.table.current_round_pot
                    self.assertAlmostEqual(sum(p.stack for p in session.game.table.players) + pot, count * 20)
                session.finish_hand()
                busted = {seat for seat, player in enumerate(session.game.table.players) if player.stack == 0}
                if session.match_finished:
                    break
                session.advance()
            self.assertFalse(session.match_truncated)
            self.assertEqual(len(busted), count - 1)
            self.assertEqual(session.stats.matches_completed, 1)
            self.assertEqual(sum(session.stats.wins), 1)
            winner = session.match_winner
            self.assertEqual(session.stats.wins[winner], 1)
            self.assertAlmostEqual(session.game.table.players[winner].stack, count * 20)
            session.advance()
            self.assertEqual(session.match, 2)
            self.assertEqual(session.hand_in_match, 1)
            self.assertEqual(session.game.hand_start_stacks, (20,) * count)

    def test_batch_target_counts_matches_not_hands(self):
        session = self.make_session(5, defer_callbacks=True, game_kwargs={"initial_stacks": 20})
        for _ in range(200):
            if session.stats.matches >= 4:
                break
            session.run_batch(4, max_actions=21)
        self.assertEqual(session.stats.matches_completed, 4)
        self.assertEqual(sum(session.stats.wins), 4)
        self.assertGreater(session.stats.hands, 4)
        stats = session.stats
        session.run_batch(4)
        self.assertEqual(session.stats, stats)

    def test_limit_interrupts_a_match_without_inventing_a_winner_and_inference_keeps_agents(self):
        session = TableSession(Registry(), [("learner", {})] * 3, continuous=True,
                               max_hands_per_match=2, mode="train", training_seats=(0,))
        self.addCleanup(session.close)
        session.run_batch(1)
        self.assertEqual(session.stats.hands, 2)
        self.assertEqual(session.stats.matches_truncated, 1)
        self.assertEqual(session.stats.matches_completed, 0)
        self.assertEqual(sum(session.stats.wins), 0)
        self.assertIsNone(session.match_winner)
        terminal = [t for t in session.agents[0].transitions if t.terminated]
        self.assertFalse(terminal[0].info["match_finished"])
        self.assertTrue(terminal[1].info["match_finished"])
        identities = [id(agent) for agent in session.agents]
        updates = len(session.agents[0].transitions)
        session.use_inference()
        self.assertEqual(session.match, 2)
        self.assertEqual(session.game.hand_start_stacks, (100,) * 3)
        session.run_batch(2)
        self.assertEqual([id(agent) for agent in session.agents], identities)
        self.assertEqual(len(session.agents[0].transitions), updates)
        self.assertTrue(all(not agent.flags[-1] for agent in session.agents))


if __name__ == "__main__":
    unittest.main()
