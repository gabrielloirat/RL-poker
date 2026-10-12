from dataclasses import dataclass
import random

from tools.hand_evaluator import get_winners
from tools.montecarlo_python import get_equity


@dataclass(frozen=True)
class EquitySnapshot:
    hands: tuple[tuple[str, ...], ...]
    board: tuple[str, ...]
    eligible: tuple[bool, ...]
    terminated: bool

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
