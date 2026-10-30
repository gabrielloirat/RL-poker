
import random

from tools.hand_evaluator import _calc_score

VALUES = "23456789TJQKA"
SUITS = "CDHS"
FULL_DECK = tuple(v + s for v in VALUES for s in SUITS)


def get_equity(player_cards, table_cards, players, runs=1000, *, rng=None):


    hero = list(player_cards)
    board = list(table_cards)
    players = max(2, int(players))
    runs = max(1, int(runs))
    sampler = rng if rng is not None else random

    known = set(hero + board)
    if len(known) != len(hero) + len(board):
        raise ValueError("Duplicate known card")

    available = [card for card in FULL_DECK if card not in known]
    needed = (players - 1) * 2 + (5 - len(board))
    if needed > len(available):
        raise ValueError("Not enough cards for simulation")

    equity = 0.0
    for _ in range(runs):
        drawn = sampler.sample(available, needed)
        offset = 0
        opponents = []
        for _ in range(players - 1):
            opponents.append(drawn[offset:offset + 2])
            offset += 2
        completed_board = board + drawn[offset:]

        hands = [hero + completed_board] + [opp + completed_board for opp in opponents]
        scores = [_calc_score(hand) for hand in hands]
        best = max(scores)
        winners = [i for i, score in enumerate(scores) if score == best]
        if 0 in winners:
            equity += 1.0 / len(winners)

    return equity / runs
