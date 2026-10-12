import numpy as np


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, action_mask, *, training=False):
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("L'agent aléatoire doit avoir au moins une action légale.")
        return int(self.rng.choice(legal))
