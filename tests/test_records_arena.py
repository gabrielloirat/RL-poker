from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from poker_rl import AgentRegistry
from poker_rl.arena import run_arena
from poker_rl.table_session import TableSession
from run_agents import main
from poker_rl.records import write_json


SAVING_AGENT = '''
import json
from pathlib import Path
class SavingAgent:
    def __init__(self, seed=None, aggressive=True):
        self.updates = 0
        self.aggressive = aggressive
    def act(self, observation, action_mask, *, training=False):
        if self.aggressive and action_mask[7]:
            return 7
        return 1 if action_mask[1] else 2
    def observe(self, transition):
        self.updates += 1
    def save_checkpoint(self, path):
        Path(path).write_text(json.dumps({'updates': self.updates, 'aggressive': self.aggressive}))
    def load_checkpoint(self, path):
        data = json.loads(Path(path).read_text())
        self.updates = data['updates']
        self.aggressive = data['aggressive']
'''


class RecordsArenaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plugins = self.root / "agents"
        self.plugins.mkdir()
        (self.plugins / "learner.py").write_text(SAVING_AGENT, encoding="utf-8")
        self.records = self.root / "records"
        self.registry = AgentRegistry(self.plugins, records_dir=self.records)

    def train(self, count=2):
        session = TableSession(self.registry, [("learner", {})] * count,
            mode="train", training_seats=(0,), record=True, game_kwargs={"initial_stacks": 20})
        self.addCleanup(session.close)
        session.run_batch(5)
        path = session.finish_recording()
        return session, path

    def test_archive_round_trip_and_full_hand_history(self):
        session, path = self.train()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual([a["training"] for a in manifest["agents"]], [True, False])
        self.assertEqual(manifest["environment"]["big_blind"], 2)
        self.assertEqual(manifest["stats"]["completed"], 5)
        history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        self.assertEqual(sum(item["event"] == "hand_start" for item in history), 5)
        self.assertEqual(sum(item["event"] == "hand_end" for item in history), 5)
        actions = [item for item in history if item["event"] == "action"]
        self.assertEqual(len(actions), sum(session.stats.actions))
        self.assertTrue(all(item["action"] in item["before"]["legal_actions"] for item in actions))
        self.assertEqual(len(history[0]["state"]["hole_cards"]), 2)
        catalog = AgentRegistry(self.plugins, records_dir=self.records)
        name = manifest["agents"][0]["model_id"]
        restored = catalog.load(name)
        self.assertEqual(restored.updates, session.agents[0].updates)
        self.assertGreater(restored.updates, 0)
        self.assertEqual(catalog.describe(name)["source_run"], manifest["id"])
        self.assertEqual(catalog.describe(name)["name"], "learner")

    def test_arena_counts_complete_matches_and_does_not_train(self):
        _, path = self.train()
        catalog = AgentRegistry(self.plugins, records_dir=self.records)
        source = json.loads((path / "manifest.json").read_text())
        names = [a["model_id"] for a in source["agents"]]
        before = [catalog.load(name).updates for name in names]
        report = run_arena(catalog, [(name, {}) for name in names], matches=10,
                           game_kwargs={"initial_stacks": 20})
        self.assertEqual(report["matches_completed"], 10)
        self.assertEqual(sum(p["wins"] for p in report["players"]), 10)
        self.assertEqual(report["matches_truncated"], 0)
        self.assertEqual(report["transitions"], (0, 0))
        for player in report["players"]:
            self.assertEqual(player["wins"] + player["losses"], 10)
        archive = Path(report["archive"])
        manifest = json.loads((archive / "manifest.json").read_text())
        after_catalog = AgentRegistry(self.plugins, records_dir=self.records)
        after = [after_catalog.load(a["model_id"]).updates for a in manifest["agents"]]
        self.assertEqual(after, before)
        self.assertTrue((archive / "arena.json").is_file())
        self.assertEqual([a["source_model"] for a in manifest["agents"]], names)

    def test_truncated_matches_have_no_invented_winner(self):
        selections = [("learner", {"aggressive": False})] * 3
        report = run_arena(self.registry, selections, matches=4, max_hands_per_match=1)
        self.assertEqual(report["matches_completed"], 0)
        self.assertEqual(report["matches_truncated"], 4)
        self.assertTrue(all(p["wins"] == p["losses"] == 0 and p["win_rate"] is None
                            for p in report["players"]))

    def test_table_size_of_saved_models_is_checked(self):
        _, path = self.train(count=3)
        manifest = json.loads((path / "manifest.json").read_text())
        catalog = AgentRegistry(self.plugins, records_dir=self.records)
        with self.assertRaisesRegex(ValueError, "3 joueurs"):
            TableSession(catalog, [(manifest["agents"][0]["model_id"], {}), ("learner", {})])

    def test_corrupt_checkpoint_rejected_and_broken_manifest_ignored(self):
        _, path = self.train()
        manifest = json.loads((path / "manifest.json").read_text())
        entry = manifest["agents"][0]
        (path / entry["checkpoint"]).write_text("broken")
        broken = self.records / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("{")
        catalog = AgentRegistry(self.plugins, records_dir=self.records)
        with self.assertRaisesRegex(ValueError, "corrompu"):
            catalog.load(entry["model_id"])

    def test_history_still_saved_without_agent_serialization_hooks(self):
        (self.plugins / "plain.py").write_text(SAVING_AGENT.split("    def save_checkpoint")[0])
        registry = AgentRegistry(self.plugins, records_dir=self.records)
        session = TableSession(registry, [("plain", {})] * 2, mode="train", training_seats=(0,), record=True)
        self.addCleanup(session.close)
        session.run_batch(2)
        path = session.finish_recording()
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertTrue(all(a["checkpoint_status"] == "unsupported" for a in manifest["agents"]))
        self.assertEqual(manifest["stats"]["completed"], 2)

    def test_cli_train_models_and_arena(self):
        common = ["--agents-dir", str(self.plugins), "--records-dir", str(self.records)]
        with redirect_stdout(io.StringIO()):
            trained = main(["train", *common, "--agent0", "learner", "--agent1", "learner", "--episodes", "3", "--validation-hands", "8"])
            catalog = main(["models", *common])
            names = list(catalog.saved_models)
            self.assertEqual(len(names), 1)  # The untrained opponent is not another bot.
            version = main(["save", *common, "--model", names[0], "--name", "Version de test"])
            metrics = main(["metrics", *common, "--model", version])
            self.assertEqual(metrics["status"], "completed")
            report = main(["arena", *common, "--agent0", names[0], "--agent1", version, "--matches", "4"])
        self.assertTrue(Path(trained["archive"]).is_dir())
        self.assertEqual(report["matches_completed"], 4)

    def test_stop_and_resume_create_separate_archives(self):
        session, first = self.train()
        session.use_inference()
        session.run_batch(7)
        second = session.finish_recording()
        self.assertNotEqual(first, second)
        manifest = json.loads((second / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "infer")
        self.assertTrue(all(not a["training"] for a in manifest["agents"]))
        self.assertEqual(manifest["stats"]["completed"], 2)

    def test_cancelled_arena_is_archived_without_fake_results(self):
        stop = Event()
        stop.set()
        report = run_arena(self.registry, [("learner", {})] * 2, matches=10, stop_event=stop)
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["matches_completed"], 0)

    def test_random_and_neural_checkpoints_restore_state(self):
        import torch
        from agents.ppo import PPOAgent
        from agents.random import RandomAgent
        from poker_rl import PokerEnv
        mask = np.array([True, True, True, False, False, False, False, False])
        random = RandomAgent(seed=7)
        target = self.root / "random.checkpoint"
        random.save_checkpoint(target)
        expected = [random.act(None, mask) for _ in range(30)]
        restored = RandomAgent()
        restored.load_checkpoint(target)
        self.assertEqual(expected, [restored.act(None, mask) for _ in range(30)])
        obs = np.zeros(147, dtype=np.float32)
        agent = PPOAgent(seed=7, autoload=False, deterministic=True)
        env = PokerEnv(opponent=RandomAgent(seed=8))
        self.addCleanup(env.close)
        agent.ppo_kwargs.update(n_steps=8, batch_size=4, n_epochs=1)
        agent.learn(env, total_timesteps=8)
        agent.act(obs, mask)
        with torch.no_grad():
            agent.model.policy.action_net.bias[1] = 4.0
        target = self.root / "ppo.checkpoint"
        agent.save_checkpoint(target)
        loaded = PPOAgent(checkpoint=target)
        self.assertTrue(all(torch.equal(value, loaded.model.policy.state_dict()[key])
                            for key, value in agent.model.policy.state_dict().items()))
        self.assertEqual(agent.act(obs, mask), loaded.act(obs, mask))

    def test_atomic_manifest_write_retries_temporary_windows_locks(self):
        target = self.root / "manifest.json"
        target.write_text('{"old":true}')
        original = Path.replace
        attempts = []
        def locked_once(source, destination):
            attempts.append(source)
            if len(attempts) == 1:
                raise PermissionError("file temporarily locked")
            return original(source, destination)
        with patch.object(Path, "replace", locked_once), patch("poker_rl.records.time.sleep"):
            write_json(target, {"new": True})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(json.loads(target.read_text()), {"new": True})


if __name__ == "__main__":
    unittest.main()
