"""CPU batch-one equivalent of fixed poker features, for benchmark processes.

Gradient updates keep the original torch path. This avoids tiny torch CPU
dispatches per inference; no learned layer, policy or game rule is changed.
"""
import numpy as np
import torch
from functools import lru_cache, wraps
import sys


def cpu_features(extractor, observations):
    obs = observations.detach().numpy()
    raw = obs.copy()
    raw[..., extractor.fields['seat']] = 0
    hole = obs[..., :52].reshape(-1, 13, 4)
    cards = hole + obs[..., 52:104].reshape(-1, 13, 4)
    hole_ranks, ranks = hole.sum(-1), cards.sum(-1)
    hole_suits, suits = hole.sum(-2), cards.sum(-2)
    present = ranks > 0

    def straight(values):
        result = values[..., [12, 0, 1, 2, 3]].sum(-1) >= 5
        for start in range(9):
            result |= values[..., start:start + 5].sum(-1) >= 5
        return result

    pairs, trips = (ranks >= 2).sum(-1), (ranks >= 3).sum(-1)
    conditions = (pairs >= 1, pairs >= 2, trips >= 1, straight(present),
                  (suits >= 5).any(-1), (trips >= 1) & (pairs >= 2),
                  (ranks >= 4).any(-1), straight(cards.swapaxes(-1, -2) > 0).any(-1))
    category = np.zeros_like(pairs)
    for value, condition in enumerate(conditions, 1):
        category[condition] = value
    combination = np.eye(9, dtype=np.float32)[category]
    high = np.where(hole_ranks > 0, np.arange(1, 14, dtype=np.float32), np.float32(0)).max(-1) / np.float32(13)
    low = np.minimum(np.where(hole_ranks > 0, np.arange(1, 14, dtype=np.float32), np.float32(14)).min(-1), np.float32(13)) / np.float32(13)
    pot = obs[..., extractor.fields['pots']].sum(-1)
    call = obs[..., extractor.fields['to_call']].squeeze(-1)
    stack = obs[..., extractor.fields['stacks']][..., 0]
    connected = np.stack([present[..., start:start + 5].sum(-1) for start in range(9)], -1).max(-1).astype(np.float32)
    scalars = np.stack((high, low, (hole_ranks >= 2).any(-1).astype(np.float32),
        (hole_suits >= 2).any(-1).astype(np.float32), np.abs(high - low),
        call / np.maximum(pot + call, np.float32(1e-8)),
        call / np.maximum(stack, np.float32(1e-8)), connected / np.float32(5)), -1)
    result = np.concatenate((raw, hole_ranks / np.float32(2), ranks / np.float32(4),
        np.sort(hole_suits, axis=-1) / np.float32(2), np.sort(suits, axis=-1) / np.float32(7),
        combination, scalars), -1)
    return torch.from_numpy(result)


def install_fast_features():
    from poker_rl.features import PokerFeaturesExtractor
    if getattr(PokerFeaturesExtractor, '_benchmark_fast_installed', False):
        return
    original = PokerFeaturesExtractor.forward

    def forward(self, observations):
        if not torch.is_grad_enabled() and observations.device.type == 'cpu' and observations.shape[0] == 1:
            return cpu_features(self, observations)
        return original(self, observations)

    PokerFeaturesExtractor.forward = forward
    PokerFeaturesExtractor._benchmark_fast_installed = True


def install_strength_cache(registry):
    """Memoize a pure card-strength function; never cache searches or actions."""
    module = sys.modules[registry.agents['mcts'].factory.__module__]
    module.MCTSPolicy._benchmark_weight_factors = policy_weight_factors
    original = module.evaluate_hand_strength
    if getattr(original, '_benchmark_cached', False):
        return original

    @lru_cache(maxsize=65536)
    def cached(hole, board):
        return original(hole, board)

    @wraps(original)
    def evaluate(hole, board):
        return cached(tuple(hole), tuple(board))

    evaluate._benchmark_cached = True
    evaluate.cache_info = cached.cache_info
    evaluate.cache_clear = cached.cache_clear
    module.evaluate_hand_strength = evaluate
    return evaluate


@lru_cache(maxsize=1024)
def cached_weight_factors(dtype, row_bytes):
    # Keep the exact scalar NumPy operations and dtypes of the original policy.
    values = np.frombuffer(row_bytes, dtype=np.dtype(dtype))
    return tuple(np.exp(np.clip(value, -3.0, 3.0)) for value in values)


def policy_weight_factors(policy, bucket):
    row = policy.weights[bucket]
    return cached_weight_factors(row.dtype.str, row.tobytes())
