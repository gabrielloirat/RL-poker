import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from poker_rl.game import observation_fields


class PokerFeaturesExtractor(BaseFeaturesExtractor):
    EXTRA_SIZE = 13 + 13 + 4 + 4 + 9 + 8

    def __init__(self, observation_space):  # Repère les champs de l'observation
        size = observation_space.shape[0]
        super().__init__(observation_space, size + self.EXTRA_SIZE)
        for players in range(2, 6):
            fields = observation_fields(players)
            total = 0
            for name, length in fields:
                total += length
            if total == size:
                self.fields = {}
                offset = 0
                for name, length in fields:
                    self.fields[name] = slice(offset, offset + length)
                    offset += length
                break
        else:
            raise ValueError("Schéma d'observation poker inconnu.")
        self.register_buffer("ranks", torch.arange(13, dtype=torch.float32))

    def _straight(self, present):  # Repère une suite de cinq cartes
        straight = present[..., [12, 0, 1, 2, 3]].sum(-1) >= 5
        for start in range(9):
            straight = straight | (present[..., start:start + 5].sum(-1) >= 5)
        return straight

    def forward(self, observations):  # Ajoute les informations sur la main
        raw = observations.clone()
        raw[..., self.fields["seat"]] = 0
        hole = observations[..., :52].reshape(-1, 13, 4)
        board = observations[..., 52:104].reshape(-1, 13, 4)
        cards = hole + board
        hole_ranks = hole.sum(-1)
        ranks = cards.sum(-1)
        present = ranks > 0
        hole_suits = hole.sum(-2)
        suits = cards.sum(-2)

        pairs = (ranks >= 2).sum(-1)
        trips = (ranks >= 3).sum(-1)
        quads = (ranks >= 4).any(-1)
        straight = self._straight(present.float())
        flush = (suits >= 5).any(-1)
        straight_flush = self._straight((cards.transpose(-1, -2) > 0).float()).any(-1)
        conditions = [
            pairs >= 1, pairs >= 2, trips >= 1, straight, flush,
            (trips >= 1) & (pairs >= 2), quads, straight_flush,
        ]
        category = torch.zeros_like(pairs)
        for value, condition in enumerate(conditions, start=1):
            category[condition] = value
        combination = torch.nn.functional.one_hot(category, 9).float()

        high = torch.where(hole_ranks > 0, self.ranks + 1, 0).amax(-1) / 13
        low = torch.where(hole_ranks > 0, self.ranks + 1, 14).amin(-1).clamp(max=13) / 13
        pot = observations[..., self.fields["pots"]].sum(-1)
        call = observations[..., self.fields["to_call"]].squeeze(-1)
        stack = observations[..., self.fields["stacks"]][..., 0]
        scalars = torch.stack((
            high, low,
            (hole_ranks >= 2).any(-1).float(),
            (hole_suits >= 2).any(-1).float(),
            (high - low).abs(),
            call / (pot + call).clamp(min=1e-8),
            call / stack.clamp(min=1e-8),
            present.float().unfold(-1, 5, 1).sum(-1).amax(-1) / 5,
        ), dim=-1)
        return torch.cat((
            raw, hole_ranks / 2, ranks / 4,
            hole_suits.sort(-1).values / 2,
            suits.sort(-1).values / 7,
            combination, scalars,
        ), dim=-1)
