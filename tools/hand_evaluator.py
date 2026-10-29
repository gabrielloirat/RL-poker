CARD_RANKS_ORIGINAL = "23456789TJQKA"
SUITS_ORIGINAL = "CDHS"


def get_winners(player_hands, table_cards):

    scores = [_calc_score(hand + table_cards) for hand in player_hands]
    best = max(score[:2] for score in scores)
    winners = [i for i, score in enumerate(scores) if score[:2] == best]
    return winners, scores[winners[0]][2]


def _calc_score(hand):
    rcounts = {CARD_RANKS_ORIGINAL.find(r): "".join(hand).count(r) for r, _ in hand}.items()
    score, card_ranks = zip(*sorted((cnt, rank) for rank, cnt in rcounts)[::-1])

    potential_threeofakind = score[0] == 3
    potential_twopair = score == (2, 2, 1, 1, 1)
    potential_pair = score == (2, 1, 1, 1, 1, 1)

    if score[0:2] == (3, 2) or score[0:2] == (3, 3):
        card_ranks = (card_ranks[0], card_ranks[1])
        score = (3, 2)
    elif score[0:4] == (2, 2, 2, 1):
        score = (2, 2, 1)
        kicker = max(card_ranks[2], card_ranks[3])
        card_ranks = (card_ranks[0], card_ranks[1], kicker)
    elif score[0] == 4:
        score = (4,)
        card_ranks = (card_ranks[0], max(card_ranks[1:]))
    elif len(score) >= 5:
        straight = False
        if 12 in card_ranks:
            card_ranks += (-1,)
        sorted_card_ranks = sorted(card_ranks, reverse=True)
        for i in range(len(sorted_card_ranks) - 4):
            straight = sorted_card_ranks[i] - sorted_card_ranks[i + 4] == 4
            if straight:
                card_ranks = tuple(sorted_card_ranks[i:i + 5])
                break

        suits = [s for _, s in hand]
        flush = max(suits.count(s) for s in suits) >= 5
        if flush:
            flush_suit = next(s for s in SUITS_ORIGINAL if suits.count(s) >= 5)
            flush_hand = [k for k in hand if flush_suit in k]
            rcounts_flush = {CARD_RANKS_ORIGINAL.find(r): "".join(flush_hand).count(r) for r, _ in flush_hand}.items()
            score, card_ranks = zip(*sorted((cnt, rank) for rank, cnt in rcounts_flush)[::-1])
            card_ranks = tuple(sorted(card_ranks, reverse=True))
            if 12 in card_ranks and -1 not in card_ranks:
                card_ranks += (-1,)
            straight = False
            for i in range(len(card_ranks) - 4):
                straight = card_ranks[i] - card_ranks[i + 4] == 4
                if straight:
                    card_ranks = tuple(card_ranks[i:i + 5])
                    break

        score = ([(1,), (3, 1, 2)], [(3, 1, 3), (5,)])[flush][straight]

    if score == (1,) and potential_threeofakind:
        score = (3, 1)
    elif score == (1,) and potential_twopair:
        score = (2, 2, 1)
    elif score == (1,) and potential_pair:
        score = (2, 1, 1)

    if score[0] == 5:
        hand_type = "StraightFlush"
    elif score[0] == 4:
        hand_type = "FourOfAKind"
    elif score[0:2] == (3, 2):
        hand_type = "FullHouse"
    elif score[0:3] == (3, 1, 3):
        hand_type = "Flush"
        card_ranks = card_ranks[:5]
    elif score[0:3] == (3, 1, 2):
        hand_type = "Straight"
        card_ranks = card_ranks[:5]
    elif score[0:2] == (3, 1):
        hand_type = "ThreeOfAKind"
        card_ranks = card_ranks[:3]
    elif score[0:2] == (2, 2):
        hand_type = "TwoPair"
        card_ranks = card_ranks[:3]
    elif score[0] == 2:
        hand_type = "Pair"
        card_ranks = card_ranks[:4]
    elif score[0] == 1:
        hand_type = "HighCard"
        card_ranks = card_ranks[:5]
    else:
        raise RuntimeError("Card type error")

    return score, card_ranks, hand_type
