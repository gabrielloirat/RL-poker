import numpy as np
import json
from pathlib import Path


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def save_checkpoint(self, path):
        Path(path).write_text(json.dumps(self.rng.bit_generator.state), encoding="utf-8")

    def load_checkpoint(self, path):
        self.rng.bit_generator.state = json.loads(Path(path).read_text(encoding="utf-8"))

    def act(self, observation, action_mask, *, training=False):
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("L'agent aléatoire doit avoir au moins une action légale.")
        return int(self.rng.choice(legal))
