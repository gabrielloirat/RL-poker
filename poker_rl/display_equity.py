import random

from tools.hand_evaluator import get_winners
from tools.montecarlo_python import get_equity


class EquitySnapshot:
    def __init__(self,
                 hands, board, eligible,
                 terminated):  # Enregistre les données
        self.hands = hands
        self.board = board
        self.eligible = eligible
        self.terminated = terminated

    def __eq__(self, other):  # Compare les valeurs
        if not isinstance(other, EquitySnapshot):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):  # Identifie un état dans le cache
        return hash((self.hands, self.board, self.eligible, self.terminated))

    @classmethod
    def from_game(cls, game):
        table = game.table
        return cls(tuple(tuple(player.cards) for player in table.players),
                   tuple(table.table_cards),
                   tuple(bool(value) for value in table.player_cycle.get_potential_winners()),
                   game.terminated)


def estimate_equities(snapshot, *, runs=1000):


    live = [seat for seat, eligible in enumerate(snapshot.eligible) if eligible]
    values = [0.0] * len(snapshot.hands)
    if len(live) == 1:
        values[live[0]] = 1.0
    elif snapshot.terminated and len(snapshot.board) == 5:
        winners, _ = get_winners([list(snapshot.hands[seat]) for seat in live], list(snapshot.board))
        for winner in winners:
            values[live[winner]] = 1.0 / len(winners)
    elif live:
        rng = random.Random()
        for seat in live:
            values[seat] = get_equity(snapshot.hands[seat], snapshot.board, len(live), runs, rng=rng)
    return tuple(values)
