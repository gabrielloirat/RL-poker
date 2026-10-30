"""Descriptive poker statistics. Outcomes never label an action as optimal."""
from collections import defaultdict

RANKS = 'AKQJT98765432'


def starting_hand(cards):
    if len(cards) != 2:
        return None
    ranks = [card[0] for card in cards]
    ranks.sort(key=RANKS.index)
    return ''.join(ranks) + ('' if ranks[0] == ranks[1] else
                             's' if cards[0][1] == cards[1][1] else 'o')


def matrix_key(row, column):
    a, b = RANKS[row], RANKS[column]
    return a + b if row == column else a + b + 's' if row < column else b + a + 'o'


def position_name(seat, dealer, count):
    offset = (seat - dealer) % count
    if count == 2:
        return ('BTN/SB', 'BB')[offset]
    return {0: 'BTN', 1: 'SB', 2: 'BB'}.get(offset,
        'CO' if offset == count - 1 else 'UTG' if offset == 3 else f'UTG+{offset - 3}')


def _rate(numerator, denominator):
    return {'numerator': int(numerator), 'denominator': int(denominator),
            'rate': numerator / denominator if denominator else None}


def _stats(hands):
    vpip = pfr = eligible = three = three_opportunities = 0
    big_folds = big_faced = 0
    streets = defaultdict(lambda: [0, 0])
    for hand in hands:
        own = [a for a in hand.get('actions', []) if a['seat'] == hand['bot_seat']]
        pre = [a for a in own if a['stage'] == 'PREFLOP']
        eligible += bool(pre)
        def raises(a):
            return a.get('is_raise', a['action'].startswith('RAISE') or
                         (a['action'] == 'ALL_IN' and a.get('stack', 0) > a.get('to_call', 0)))
        vpip += any(a.get('voluntary', raises(a) or
                     (a['action'] in ('CALL', 'ALL_IN') and a.get('to_call', 0) > 0)) for a in pre)
        pfr += any(raises(a) for a in pre)
        # A 3-bet opportunity is a decision facing exactly one previous raise.
        opportunities = [a for a in pre if a.get('preflop_raises') == 1 and a.get('can_raise', False)]
        three_opportunities += bool(opportunities)
        three += any(raises(a) for a in opportunities)
        for a in own:
            # Pot includes the outstanding bet. Equivalent to a pot-size bet
            # heads-up when the caller has contributed nothing to that bet.
            if a.get('to_call', 0) > 0 and a.get('pot', 0) > 0 and a['to_call'] / a['pot'] >= .5:
                big_faced += 1
                folded = a['action'] == 'FOLD'
                big_folds += folded
                streets[a['stage']][0] += folded
                streets[a['stage']][1] += 1
    net = sum(h['net_bb'] for h in hands)
    return {'hands': len(hands), 'net_bb': net,
            'bb_per_100': 100 * net / len(hands) if hands else None,
            'vpip': _rate(vpip, eligible), 'pfr': _rate(pfr, eligible),
            'three_bet': _rate(three, three_opportunities),
            'fold_large_bet': _rate(big_folds, big_faced),
            'fold_large_bet_by_street': {k: _rate(*v) for k, v in streets.items()}}


def analyze_strategy(hands):
    complete = [h for h in hands if h.get('terminated')]
    groups = {matrix_key(r, c): [] for r in range(13) for c in range(13)}
    positions = defaultdict(list)
    for hand in complete:
        key = hand.get('starting_hand') or starting_hand(hand.get('hole_cards', []))
        if key in groups:
            groups[key].append(hand)
        position = hand.get('position') or position_name(hand['bot_seat'], hand['dealer'], len(hand['player_cards']))
        positions[position].append(hand)
    return {'version': 1, 'overall': _stats(complete),
            'starting_hands': {k: _stats(v) for k, v in groups.items()},
            'positions': {k: _stats(v) for k, v in positions.items()},
            'definitions': {'vpip_pfr': 'Hands with a preflop decision; blinds excluded; all-in calls are not raises.',
                'three_bet': 'Hands with a legal raise opportunity facing exactly one prior preflop raise.',
                'large_bet': 'Decision with to_call / pot_before_action >= 0.5; pot includes outstanding bets.',
                'outcomes': 'Completed hands only; associations, not counterfactual action quality.'}}
