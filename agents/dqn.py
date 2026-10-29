from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

NUM_ACTIONS = 8


class QNetwork(nn.Module):
    """Perceptron multicouche : observation -> 8 valeurs Q (une par action)."""

    def __init__(self, input_dim: int, hidden_sizes=(256, 256), num_actions: int = NUM_ACTIONS):
        super().__init__()
        layers, last = [], input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Mémoire circulaire de transitions (s, mask, a, r, s', mask', done)."""

    def __init__(self, capacity: int, input_dim: int):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.obs = np.zeros((capacity, input_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, NUM_ACTIONS), dtype=np.bool_)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, input_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, NUM_ACTIONS), dtype=np.bool_)
        self.done = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, obs, mask, action, reward, next_obs, next_mask, done):
        i = self.pos
        self.obs[i], self.mask[i], self.action[i], self.reward[i] = obs, mask, action, reward
        self.next_obs[i], self.next_mask[i], self.done[i] = next_obs, next_mask, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.mask[idx]),
                torch.as_tensor(self.action[idx]), torch.as_tensor(self.reward[idx]),
                torch.as_tensor(self.next_obs[idx]), torch.as_tensor(self.next_mask[idx]),
                torch.as_tensor(self.done[idx]))


class DQNAgent:
    def __init__(self, checkpoint=None, hidden_sizes=(256, 256), gamma=0.99,
                 learning_rate=1e-3, batch_size=64, buffer_size=100_000,
                 learning_starts=1_000, train_every=4, target_update=500,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50_000,
                 reward_scale=50.0, checkpoint_every=500, seed=None, name="DQN"):
        self.name = name
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.reward_scale = reward_scale          # 50 BB = gain max à 2 joueurs (stack 100, BB 2)
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = Path(checkpoint) if checkpoint else None

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.input_dim = None
        self.online = self.target = self.optimizer = self.buffer = None
        self.act_steps = 0       # décisions prises en entraînement (pilote epsilon)
        self.learn_steps = 0     # mises à jour du réseau
        self.episodes = 0
        self.last_loss = None

        if self.checkpoint_path and self.checkpoint_path.is_file():
            self.load_checkpoint(self.checkpoint_path)

    # ------------------------------------------------------------------ réseau
    def _build(self, input_dim: int):
        self.input_dim = input_dim
        self.online = QNetwork(input_dim, self.hidden_sizes)
        self.target = QNetwork(input_dim, self.hidden_sizes)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(self.buffer_size, input_dim)

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.act_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------ contrat
    def act(self, observation, action_mask, *, training=False):
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("L'agent DQN doit avoir au moins une action légale.")
        if self.online is None:
            self._build(len(observation))
        if training:
            explore = self.rng.random() < self.epsilon
            self.act_steps += 1
            if explore:
                return int(self.rng.choice(list(legal)))
        with torch.no_grad():
            q = self.online(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0].numpy()
        q = np.where(action_mask, q, -np.inf)       # jamais d'action illégale
        return int(np.argmax(q))

    def observe(self, transition):
        if self.online is None:
            self._build(len(transition.observation))
        done = bool(transition.terminated or transition.truncated)
        self.buffer.push(transition.observation, transition.action_mask, transition.action,
                         float(transition.reward) / self.reward_scale,
                         transition.next_observation, transition.next_action_mask, done)
        if len(self.buffer) >= max(self.learning_starts, self.batch_size) \
                and self.act_steps % self.train_every == 0:
            self._learn()

    def on_episode_start(self, *, seed=None, training=False):
        pass

    def on_episode_end(self, *, reward=0.0, terminated=False, truncated=False, training=False):
        if training:
            self.episodes += 1
            if self.checkpoint_path and self.episodes % self.checkpoint_every == 0:
                self.save_checkpoint(self.checkpoint_path)

    # ------------------------------------------------------------------ apprentissage
    def _learn(self):
        obs, mask, action, reward, next_obs, next_mask, done = self.buffer.sample(self.batch_size, self.np_rng)

        q = self.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN : le réseau en ligne CHOISIT l'action (parmi les légales),
            # le réseau cible l'ÉVALUE.
            next_online = self.online(next_obs).masked_fill(~next_mask, -1e9)
            best = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best).squeeze(1)
            target = reward + self.gamma * (1.0 - done) * next_q

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        self.last_loss = float(loss.item())
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path):
        if self.online is None:
            return
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 1, "input_dim": self.input_dim, "hidden_sizes": list(self.hidden_sizes),
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "act_steps": self.act_steps, "learn_steps": self.learn_steps, "episodes": self.episodes,
        }, file_path)

    def load_checkpoint(self, path):
        data = torch.load(Path(path), map_location="cpu")
        self.hidden_sizes = tuple(data["hidden_sizes"])
        self._build(int(data["input_dim"]))
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.act_steps = int(data.get("act_steps", 0))
        self.learn_steps = int(data.get("learn_steps", 0))
        self.episodes = int(data.get("episodes", 0))