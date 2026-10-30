
from importlib import import_module


class Transition:


    def __init__(self,
                 observation, action_mask, action,
                 reward, next_observation, next_action_mask,
                 terminated, truncated, info):  # Enregistre les données
        self.observation = observation
        self.action_mask = action_mask
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.next_action_mask = next_action_mask
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def to_dict(self):  # Prépare les données à sauvegarder
        return {
            "observation": self.observation,
            "action_mask": self.action_mask,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "next_action_mask": self.next_action_mask,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


class Agent:
    def act(self, observation, action_mask, training=False):  # À définir dans chaque agent
        raise NotImplementedError("L'agent doit définir act().")


class TrainableAgent(Agent):
    def observe(self, transition):  # À définir pour un agent entraînable
        raise NotImplementedError("L'agent doit définir observe().")


def load_agent(spec, kwargs = None, *,
               agents_dir = None) :

    if ":" not in spec:
        from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
        if agents_dir is None:
            agents_dir = DEFAULT_AGENTS_DIR
        registry = AgentRegistry(agents_dir)
        return registry.load(spec, kwargs)
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Agent must be specified as python_module:ClassOrFactory.")
    factory = getattr(import_module(module_name), attribute)
    agent = factory(**(kwargs or {}))
    if not callable(getattr(agent, "act", None)):
        raise TypeError(f"{spec} must provide act(observation, action_mask, *, training=False).")
    return agent
