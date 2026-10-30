import copy
import json
from pathlib import Path
import random
import tempfile
from threading import Event
import unittest

import numpy as np

from poker_rl.evaluation import evaluate_agents, validate_model, _summary
from poker_rl.plugins import AgentRegistry
from poker_rl.records import save_model_version
from poker_rl.table_session import TableSession
from test_records_arena import SAVING_AGENT


class FrozenAgent:
    def __init__(self):
        self.updates = 0
        self.memory = []

    def act(self, observation, action_mask, *, training=False):
        assert not training
        self.memory.append(len(observation))
        return 1 if action_mask[1] else 2

    def observe(self, transition):
        raise AssertionError("Validation must never deliver a learning transition")

    def on_episode_start(self, *, seed, training):
        assert not training
        random.seed(seed)
        np.random.seed(seed % 2**32)


class ValidationTests(unittest.TestCase):
    def test_frozen_policy_callbacks_random_state_and_reproducible_balanced_positions(self):
        for count in range(2, 6):
            agent = FrozenAgent()
            original = copy.deepcopy(agent.__dict__)
            random.seed(123)
            np.random.seed(456)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            report = evaluate_agents([agent] * count, (count - 1,), hands_per_opponent=count * count)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(agent.__dict__, original)
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_equal(np.random.get_state(), numpy_state)
            candidate = report["agents"][0]
            for benchmark in candidate["benchmarks"]:
                self.assertEqual(benchmark["completed"], count * count)
                self.assertEqual(benchmark["position_counts"], [count] * count)
                self.assertEqual(benchmark["wins"] + benchmark["ties"] + benchmark["losses"], count * count)
            again = evaluate_agents([agent] * count, (count - 1,), hands_per_opponent=count * count)
            self.assertEqual(candidate, again["agents"][0])

    def test_interrupted_and_truncated_validation_do_not_invent_scores(self):
        event = Event()
        event.set()
        report = evaluate_agents([FrozenAgent()] * 2, (0,), hands_per_opponent=4, stop_event=event)
        self.assertEqual(report["status"], "interrupted")
        self.assertIsNone(report["agents"][0]["benchmarks"][0]["bb_per_hand"])
        report = evaluate_agents([FrozenAgent()] * 2, (0,), hands_per_opponent=4,
                                 game_kwargs={"max_actions": 1})
        for benchmark in report["agents"][0]["benchmarks"]:
            self.assertEqual(benchmark["completed"], 0)
            self.assertEqual(benchmark["truncated"], 4)
            self.assertIsNone(benchmark["hand_win_rate"])
            self.assertIsNone(benchmark["ci95_bb_per_hand"])

    def test_statistics_have_correct_denominators_and_units(self):
        result = _summary([1., 2., 3.], 1, 1, 7, [0] * 8, [2, 1])
        self.assertEqual(result["bb_per_hand"], 2.)
        self.assertEqual(result["bb_per_100"], 200.)
        self.assertEqual(result["std_bb"], 1.)
        self.assertEqual(result["hand_win_rate"], 1 / 3)
        self.assertAlmostEqual(result["ci95_bb_per_hand"][1], 2 + 1.96 / np.sqrt(3))

    def test_copy_failure_is_reported_without_using_the_training_agent(self):
        class Uncopyable(FrozenAgent):
            def __deepcopy__(self, memo):
                raise ValueError("Cannot clone")
        agent = Uncopyable()
        report = evaluate_agents([agent, FrozenAgent()], (0,), hands_per_opponent=4)
        self.assertEqual(report["status"], "failed")
        self.assertIn("Cannot clone", report["agents"][0]["error"])
        self.assertEqual(agent.memory, [])


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.plugins = self.root / "agents"
        self.plugins.mkdir()
        (self.plugins / "learner.py").write_text(SAVING_AGENT, encoding="utf-8")
        self.records = self.root / "records"

    def registry(self):
        return AgentRegistry(self.plugins, records_dir=self.records)

    def train(self, *, count=2, hands=3):
        session = TableSession(self.registry(), [("learner", {})] * count,
            mode="train", training_seats=(0,), record=True, game_kwargs={"initial_stacks": 20})
        self.addCleanup(session.close)
        session.run_batch(hands)
        session.complete_training(validation_hands=8)
        return session

    def test_only_current_model_shown_until_explicit_save_and_versions_stay_frozen(self):
        first = self.train()
        registry = self.registry()
        self.assertEqual(len(registry.saved_models), 1)
        current = next(iter(registry.saved_models))
        self.assertEqual(registry.choices(2), [current])
        version = save_model_version(self.records, current, label="Première version")
        updates = registry.load(current).updates
        first_report = registry.saved_models[current]["validation"]
        self.train(hands=7)
        registry = self.registry()
        self.assertEqual(len(registry.saved_models), 2)
        self.assertEqual(registry.load(version).updates, updates)
        self.assertGreater(registry.load(current).updates, updates)
        self.assertEqual(registry.saved_models[version]["validation"], first_report)
        self.assertEqual(registry.label(version), "learner · Première version · 2j")
        manifest = json.loads((first.last_archive / "manifest.json").read_text())
        self.assertEqual(registry.load(manifest["agents"][0]["model_id"]).updates, updates)
        self.assertTrue((first.last_archive / "validation.json").exists())
        self.assertNotIn("validation", manifest["agents"][1])

    def test_inference_and_arena_checkpoints_do_not_create_or_replace_current_bots(self):
        session = self.train()
        before = self.registry().saved_models
        session.use_inference()
        session.run_batch(6)
        session.finish_recording()
        after = self.registry().saved_models
        self.assertEqual(list(before), list(after))
        self.assertEqual(next(iter(before.values()))["run_id"], next(iter(after.values()))["run_id"])

    def test_validation_runs_without_archiving_or_mutating_training_counts(self):
        session = TableSession(self.registry(), [("learner", {})] * 2,
            mode="train", training_seats=(0,), record=False)
        self.addCleanup(session.close)
        session.run_batch(3)
        stats, updates = session.stats, session.agents[0].updates
        self.assertIsNone(session.complete_training(validation_hands=8))
        self.assertEqual(session.validation_report["status"], "completed")
        self.assertEqual(session.stats, stats)
        self.assertEqual(session.agents[0].updates, updates)
        self.assertFalse(self.records.exists())

    def test_current_models_separated_by_observation_table_size(self):
        self.train(count=2)
        self.train(count=3)
        registry = self.registry()
        self.assertEqual(len(registry.saved_models), 2)
        self.assertEqual(len(registry.choices(2)), 1)
        self.assertEqual(len(registry.choices(3)), 1)

    def test_existing_model_validation_persists_metrics_and_preserves_checkpoint(self):
        self.train()
        registry = self.registry()
        name, model = next(iter(registry.saved_models.items()))
        before = model['path'].read_bytes()
        validation = validate_model(registry, name, hands_per_opponent=12)
        self.assertEqual(validation['status'], 'completed')
        self.assertEqual(model['path'].read_bytes(), before)
        self.assertEqual(self.registry().saved_models[name]['validation'], validation)

    def test_unloadable_old_checkpoint_has_explicit_validation_failure(self):
        from unittest.mock import patch
        self.train()
        registry = self.registry()
        name = next(iter(registry.saved_models))
        with patch.object(registry, 'load', side_effect=ValueError('Incompatible checkpoint')):
            validation = validate_model(registry, name, hands_per_opponent=12)
        self.assertEqual(validation['status'], 'failed')
        self.assertEqual(validation['benchmarks'], [])
        self.assertIn('Incompatible checkpoint', validation['error'])


if __name__ == "__main__":
    unittest.main()
