import math

import numpy as np

from poker_rl.game import ACTION_NAMES


def decision_diagnostics(probabilities, action, *, deterministic):  # Décrit une décision
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    ordered = np.sort(probabilities)
    normalized_entropy = 0.0
    if len(positive) > 1:
        normalized_entropy = entropy / math.log(len(positive))
    margin = 1.0
    if len(ordered) > 1:
        margin = float(ordered[-1] - ordered[-2])
    return {
        "probabilities": probabilities.tolist(),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "max_probability": float(ordered[-1]),
        "top_two_margin": margin,
        "chosen_probability": float(probabilities[action]),
        "greedy_action": int(np.argmax(probabilities)),
        "deterministic": bool(deterministic),
    }


def _summarize_group(decisions):  # Résume un groupe de décisions
    policies = []
    predicted = []
    for decision in decisions:
        policies.append(decision["policy"])
        predicted.append(decision["policy"]["probabilities"])
    probabilities = np.mean(predicted, axis=0)
    counts = dict.fromkeys(ACTION_NAMES, 0)
    for decision in decisions:
        if decision["action"] in counts:
            counts[decision["action"]] += 1
    summary = {
        "decisions": len(decisions),
        "mean_probabilities": dict(zip(ACTION_NAMES, probabilities.tolist())),
        "action_counts": counts,
    }
    for name in ("entropy", "normalized_entropy", "max_probability",
                 "top_two_margin", "chosen_probability"):
        values = []
        for policy in policies:
            values.append(policy[name])
        summary[name] = float(np.mean(values))
    return summary


def summarize_decisions(decisions):  # Résume les données par street
    rows = []
    streets = {}
    for decision in decisions:
        if decision.get("policy"):
            rows.append(decision)
            streets.setdefault(decision["stage"], []).append(decision)
    if not rows:
        return None
    by_street = {}
    for street, items in streets.items():
        by_street[street] = _summarize_group(items)
    return {"overall": _summarize_group(rows), "by_street": by_street}
