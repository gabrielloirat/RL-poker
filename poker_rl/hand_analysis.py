"""Descriptive analysis of complete validation hands, without learning signals."""
from gym_env.enums import Stage
from poker_rl.game import ACTION_NAMES
from tools.hand_evaluator import _calc_score
from poker_rl.strategy_metrics import analyze_strategy, starting_hand, position_name


CATEGORY_LABELS = {
    "HighCard": "Carte haute", "Pair": "Paire", "TwoPair": "Double paire",
    "ThreeOfAKind": "Brelan", "Straight": "Quinte", "Flush": "Couleur",
    "FullHouse": "Full", "FourOfAKind": "Carré", "StraightFlush": "Quinte flush",
    "Fold": "Bot couché", "NoShowdown": "Sans abattage", "Unknown": "Non classée",
}
ACTION_LABELS = dict(zip(ACTION_NAMES, (
    "Fold", "Check", "Call", "Relance 3 BB", "Relance ½ pot", "Relance pot",
    "Relance 2 pots", "Tapis")))
ACTION_LABELS["NO_DECISION"] = "Sans décision"
STREET_LABELS = {"PREFLOP": "Préflop", "FLOP": "Flop", "TURN": "Turn",
                 "RIVER": "River", "NO_DECISION": "Sans décision"}
DIMENSION_LABELS = {"combination": CATEGORY_LABELS, "action": ACTION_LABELS, "street": STREET_LABELS}


def decision_snapshot(game, seat, action):
    table = game.table
    mask = game.action_masks(seat)
    to_call = max(0, min(table.min_call - table.player_pots[seat], table.players[seat].stack))
    pot = float(table.community_pot + table.current_round_pot)
    stack = float(table.players[seat].stack)
    contributions = (0, 0, to_call, 3 * table.big_blind - table.player_pots[seat],
                     pot / 2, pot, pot * 2, stack)
    def raises(index):
        return index >= 3 and min(stack, contributions[index]) + table.player_pots[seat] > table.min_call + 1e-9
    return {"seat": seat, "stage": table.stage.name, "action": ACTION_NAMES[action],
            "is_raise": bool(raises(action)),
            "voluntary": bool(action >= 2 and min(stack, contributions[action]) > 0),
            "can_raise": bool(any(mask[index] and raises(index) for index in range(3, 8))),
            "preflop_raises": int(getattr(game, '_analysis_preflop_raises', 0)),
            "board": list(table.table_cards), "pot": float(table.community_pot + table.current_round_pot),
            "stack": float(table.players[seat].stack),
            "to_call": float(max(0, min(table.min_call - table.player_pots[seat], table.players[seat].stack)))}


def describe_hand(game, seat, *, hand, seed, actions):
    """One outcome per hand by net BB, including split-pot net losses/gains.

    Folded cards are never described as a showdown combination, even if other
    players subsequently reveal a full board. A last action is an association
    with a result, not a counterfactual assessment of that decision's quality.
    """
    table = game.table
    folded = bool(table.player_cycle.folder[seat])
    showdown = table.stage == Stage.SHOWDOWN and not folded
    category = "Unknown"
    if folded:
        category = "Fold"
    elif game.terminated and table.stage == Stage.END_HIDDEN:
        category = "NoShowdown"
    elif showdown and len(table.players[seat].cards) + len(table.table_cards) >= 5:
        category = _calc_score(list(table.players[seat].cards) + list(table.table_cards))[2]
    net = float(game.rewards()[seat])
    outcome = ("truncated" if not game.terminated else
               "win" if net > 1e-9 else "loss" if net < -1e-9 else "neutral")
    own_actions = [decision for decision in actions if decision["seat"] == seat]
    last = own_actions[-1] if own_actions else {}
    return {"hand": hand, "seed": seed, "bot_seat": seat, "dealer": table.dealer_pos,
            "position": position_name(seat, table.dealer_pos, len(table.players)),
            "starting_hand": starting_hand(table.players[seat].cards),
            "depth_bb": float(game.hand_start_stacks[seat] / table.big_blind),
            "hole_cards": list(table.players[seat].cards), "board": list(table.table_cards),
            "player_cards": [list(player.cards) for player in table.players],
            "net_bb": net, "outcome": outcome, "category": category,
            "last_action": last.get("action", "NO_DECISION"),
            "last_stage": last.get("stage", "NO_DECISION"),
            "showdown": showdown, "terminated": bool(game.terminated),
            "truncated": bool(game.truncated), "actions": actions}


def analyze_hands(hands):
    breakdown = {}
    for dimension, field in (("combination", "category"), ("action", "last_action"), ("street", "last_stage")):
        rows = {key: {"category": key, "hands": 0, "wins": 0, "losses": 0, "neutral": 0,
                      "won_bb": 0., "lost_bb": 0., "net_bb": 0.} for key in DIMENSION_LABELS[dimension]}
        for hand in hands:
            if not hand["terminated"]:
                continue
            row = rows[hand[field]]
            row["hands"] += 1
            row[{"win": "wins", "loss": "losses", "neutral": "neutral"}[hand["outcome"]]] += 1
            row["net_bb"] += hand["net_bb"]
            row["won_bb"] += max(0., hand["net_bb"])
            row["lost_bb"] += max(0., -hand["net_bb"])
        breakdown[dimension] = list(rows.values())
    return {"version": 2, "outcome_definition": "net_profit_bb", "breakdown": breakdown,
            "strategy": analyze_strategy(hands), "hands": hands}


def filter_hands(hands, *, dimension="combination", category=None, outcome="loss"):
    field = {"combination": "category", "action": "last_action", "street": "last_stage",
             "starting_hand": "starting_hand", "position": "position"}[dimension]
    def value(hand):
        if dimension == 'starting_hand':
            return hand.get(field) or starting_hand(hand.get('hole_cards', []))
        if dimension == 'position':
            return hand.get(field) or position_name(hand['bot_seat'], hand['dealer'], len(hand['player_cards']))
        return hand.get(field)
    return sorted((hand for hand in hands if hand["terminated"]
                   and (category is None or value(hand) == category)
                   and (outcome == "all" or hand["outcome"] == outcome)),
                  key=lambda hand: (hand["net_bb"] if outcome != "win" else -hand["net_bb"], hand["hand"]))


def compact_candidate(candidate, *, hands_file=None):
    """Keep the catalog small; full traces live in the separate validation report."""
    compact = {**candidate, "benchmarks": []}
    for benchmark in candidate["benchmarks"]:
        entry = dict(benchmark)
        if "hand_analysis" in entry:
            entry["hand_analysis"] = {key: value for key, value in entry["hand_analysis"].items() if key != "hands"}
            if hands_file:
                entry["hand_analysis"]["hands_file"] = hands_file
        compact["benchmarks"].append(entry)
    return compact


def compact_report(report, *, hands_file=None):
    if report is None:
        return None
    return {**report, "agents": [compact_candidate(candidate, hands_file=hands_file) for candidate in report["agents"]]}
