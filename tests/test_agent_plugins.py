from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from poker_rl import AgentRegistry, PokerEnv, load_agent, run_session
from run_agents import main


AGENT = '''
class MyAgent:
    def __init__(self, tag=None):
        self.tag = tag
        self.transitions = []
    def act(self, observation, action_mask, *, training=False):
        return 1 if action_mask[1] else 2
    def observe(self, transition):
        self.transitions.append(transition)
'''


class AgentPluginTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = self.root / "plugins"
        self.directory.mkdir()

    def write(self, path, content):
        target = self.directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def test_discovery_does_not_construct_models_and_ignores_hidden_files(self):
        self.write("model.py", AGENT.replace("self.tag = tag", "raise RuntimeError('constructor called')"))
        self.write("_helper.py", "raise AssertionError('hidden helper imported')")
        registry = AgentRegistry(self.directory)
        self.assertEqual(list(registry.agents), ["model"])
        self.assertEqual(registry.errors, {})
        with self.assertRaisesRegex(RuntimeError, "constructor called"):
            registry.load("model")

    def test_file_agents_are_independent_and_receive_training_transitions(self):
        self.write("my_agent.py", AGENT)
        registry = AgentRegistry(self.directory)
        agents = [registry.load("my_agent", {"tag": "first"}),
                  load_agent("my_agent", {"tag": "second"}, agents_dir=self.directory)]
        self.assertEqual([a.tag for a in agents], ["first", "second"])
        self.assertIsNot(agents[0], agents[1])
        results = run_session(agents, episodes=3, training_seats=(0,), seed=8)
        self.assertTrue(all(r.terminated and r.illegal_actions == 0 for r in results))
        self.assertTrue(agents[0].transitions)
        self.assertEqual(agents[1].transitions, [])

    def test_folder_agents_support_relative_imports_with_and_without_init(self):
        for name in ("plain", "package"):
            self.write(f"{name}/model.py", "TAG = 'local helper'\n")
            self.write(f"{name}/agent.py", "from .model import TAG\n" + AGENT.replace("self.tag = tag", "self.tag = TAG"))
        self.write("package/__init__.py", "")
        registry = AgentRegistry(self.directory)
        self.assertEqual(registry.errors, {})
        for name in ("plain", "package"):
            self.assertEqual(registry.load(name).tag, "local helper")

    def test_init_package_and_explicit_factory(self):
        self.write("factory/model.py", AGENT)
        self.write("factory/__init__.py", "from .model import MyAgent\ndef create_agent(**kwargs):\n    return MyAgent(**kwargs)\n")
        self.write("direct/__init__.py", AGENT)
        registry = AgentRegistry(self.directory)
        self.assertEqual(registry.load("factory", {"tag": 42}).tag, 42)
        self.assertEqual(registry.load("direct", {"tag": 24}).tag, 24)

    def test_bad_plugins_do_not_block_valid_agent(self):
        self.write("good.py", AGENT)
        self.write("broken.py", "import missing_poker_dependency_12345\n")
        self.write("ambiguous.py", AGENT + "\nclass OtherAgent(MyAgent):\n    pass\n")
        self.write("duplicate.py", AGENT)
        self.write("duplicate/agent.py", AGENT)
        registry = AgentRegistry(self.directory)
        self.assertEqual(list(registry.agents), ["good"])
        self.assertEqual(set(registry.errors), {"broken", "ambiguous", "duplicate"})
        self.assertIn("create_agent", registry.errors["ambiguous"])
        self.assertIsNotNone(registry.load("good"))
        with self.assertRaisesRegex(ValueError, "missing_poker_dependency"):
            registry.load("broken")
        with self.assertRaisesRegex(ValueError, "Disponibles : good"):
            registry.load("unknown")

    def test_factories_must_return_agent(self):
        self.write("bad.py", "def create_agent():\n    return object()\n")
        with self.assertRaisesRegex(TypeError, "act"):
            AgentRegistry(self.directory).load("bad")

    def test_namespaces_isolate_same_agent_names_in_different_directories(self):
        self.write("same.py", AGENT)
        other = self.root / "other"
        other.mkdir()
        (other / "same.py").write_text(AGENT.replace("self.tag = tag", "self.tag = 'other'"), encoding="utf-8")
        self.assertEqual(load_agent("same", {"tag": "first"}, agents_dir=self.directory).tag, "first")
        self.assertEqual(load_agent("same", agents_dir=other).tag, "other")

    def test_gym_environment_resolves_opponent_name_and_kwargs(self):
        self.write("opponent.py", AGENT)
        env = PokerEnv(opponent="opponent", agents_dir=self.directory, opponent_kwargs={"tag": 7})
        self.addCleanup(env.close)
        self.assertEqual(env.opponent.tag, 7)
        _, info = env.reset(seed=42)
        done = False
        while not done:
            action = 1 if info["action_mask"][1] else 2
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        self.assertTrue(terminated)

    def test_cli_lists_errors_and_runs_config_relative_plugins(self):
        self.write("first.py", AGENT)
        self.write("second.py", AGENT)
        self.write("broken.py", "raise RuntimeError('broken plugin')")
        output = io.StringIO()
        with redirect_stdout(output):
            main(["list", "--agents-dir", str(self.directory)])
        self.assertIn("first", output.getvalue())
        self.assertIn("broken [indisponible]", output.getvalue())
        config = self.root / "session.json"
        config.write_text(json.dumps({"agents_dir": "plugins", "episodes": 2,
                                     "agents": [{"name": "first", "kwargs": {"tag": 1}, "train": True},
                                                {"name": "second"}]}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            report = main(["train", "--config", str(config)])
            override = main(["infer", "--config", str(config), "--agent0", "second"])
        self.assertEqual(report["completed"], 2)
        self.assertEqual(override["illegal_actions"], 0)

    def test_empty_and_missing_folders(self):
        self.assertEqual(AgentRegistry(self.directory).agents, {})
        with self.assertRaisesRegex(ValueError, "Dossier d'agents introuvable"):
            AgentRegistry(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
