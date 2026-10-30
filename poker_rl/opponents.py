"""Seeded pool of automatic styles, selected independently at every hand."""
import numpy as np

class MixedOpponent:
    def __init__(self, seed=None, styles=('random', 'check_call', 'tight', 'aggressive')):
        if not styles or any(style not in ('random', 'check_call', 'tight', 'aggressive') for style in styles):
            raise ValueError('Styles attendus : random, check_call, tight, aggressive.')
        self.styles = tuple(styles)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.style = self.styles[0]

    def on_episode_start(self, *, seed=None, training=False):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, self.seed or 0]))
        self.style = self.styles[int(self.rng.integers(len(self.styles)))]

    def act(self, observation, action_mask, *, training=False):
        legal = np.flatnonzero(action_mask)
        if self.style == 'random':
            return int(self.rng.choice(legal))
        if self.style == 'check_call':
            return next((a for a in (1, 2, 0) if action_mask[a]), int(legal[0]))
        cards = np.asarray(observation[:52]).reshape(13, 4)
        ranks = cards.sum(axis=1)
        indices = np.flatnonzero(ranks)
        strong = bool(ranks.max() == 2 or (len(indices) and indices.max() >= 10))
        if self.style == 'tight':
            if strong:
                return next((a for a in (4, 3, 2, 1, 7, 0) if action_mask[a]), int(legal[0]))
            return next((a for a in (1, 0, 2) if action_mask[a]), int(legal[0]))
        # Aggression is occasional, so the pool contains both large bets and checks.
        if strong or self.rng.random() < .4:
            return next((a for a in (5, 7, 3, 2, 1, 0) if action_mask[a]), int(legal[0]))
        return next((a for a in (1, 2, 0) if action_mask[a]), int(legal[0]))
