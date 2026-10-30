"""Held-out, independent-hand validation of frozen policies after training."""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import random
import sys

import numpy as np

from poker_rl.game import PokerGame, ACTION_NAMES
from poker_rl.records import write_json
from poker_rl.hand_analysis import decision_snapshot, describe_hand, analyze_hands, compact_candidate
from poker_rl.opponents import MixedOpponent
from poker_rl.policy_diagnostics import summarize_decisions


DEFAULT_VALIDATION_HANDS = 500  # per reference opponent and trained bot
VALIDATION_SEED = 1_000_000_007


class ReferenceAgent:
    def __init__(self, kind, seed):
        self.kind, self.rng = kind, np.random.default_rng(seed)
        self.other = MixedOpponent(seed=seed, styles=(kind,)) if kind in ('tight', 'aggressive') else None

    def act(self, observation, action_mask, *, training=False):
        if self.other is not None:
            return self.other.act(observation, action_mask, training=False)
        legal = np.flatnonzero(action_mask)
        if self.kind == "random":
            return int(self.rng.choice(legal))
        return next((action for action in (1, 2, 0) if action_mask[action]), int(legal[0]))


@contextmanager
def _preserve_random_state():
    # Loading a model or a third-party callback may seed process-wide generators.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.random.get_rng_state() if torch else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)


def _summary(rewards, wins, ties, truncated, actions, positions):
    n = len(rewards)
    mean = float(np.mean(rewards)) if n else None
    std = float(np.std(rewards, ddof=1)) if n > 1 else None
    error = std / np.sqrt(n) if std is not None else None
    interval = [mean - 1.96 * error, mean + 1.96 * error] if error is not None else None
    return {"completed": n, "truncated": truncated, "net_bb": float(sum(rewards)),
            "bb_per_hand": mean, "bb_per_100": 100 * mean if mean is not None else None,
            "std_bb": std, "standard_error_bb": error, "ci95_bb_per_hand": interval,
            "wins": wins, "ties": ties, "losses": n - wins - ties,
            "hand_win_rate": wins / n if n else None,
            "actions": dict(zip(ACTION_NAMES, actions)), "position_counts": positions}


def evaluate_agents(agents, training_seats, *, game_kwargs=None,
                    hands_per_opponent=DEFAULT_VALIDATION_HANDS, seed=VALIDATION_SEED,
                    stop_event=None, on_progress=None, deterministic=None,
                    record_hands=True, opponents=('random', 'check_call'), frozen_opponents=None):
    """No observe(), no training callbacks, no shared agent state.

    All stacks reset every hand. Both the focal seat and dealer are rotated;
    opponents, cards and their RNG are independent of training. These simple
    references measure benchmark performance, not exploitability or GTO strength.
    """
    if type(hands_per_opponent) is not int or hands_per_opponent < 1:
        raise ValueError("La validation nécessite au moins une main par adversaire.")
    frozen_opponents = frozen_opponents or {}
    if not opponents or len(set(opponents)) != len(opponents) or any(
            name not in ('random', 'check_call', 'tight', 'aggressive', *frozen_opponents) for name in opponents):
        raise ValueError('Adversaires de validation inconnus ou dupliqués.')
    count = len(agents)
    settings = {**(game_kwargs or {}), "num_players": count}
    report = {"status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol": {"version": 1, "seed": seed, "hands_per_opponent": hands_per_opponent,
                           "num_players": count, "environment": settings,
                           "opponents": list(opponents), "independent_hands": True,
                           "training": False, "positions": "seat_and_dealer_rotation",
                           "policy_mode": "configured" if deterministic is None else "deterministic" if deterministic else "stochastic",
                           "ci_method": "normal_approximation_95_percent"}, "agents": []}
    with _preserve_random_state():
        for trained_seat in sorted(training_seats):
            candidate = {"seat": trained_seat, "status": "completed", "benchmarks": []}
            report["agents"].append(candidate)
            try:
                original = agents[trained_seat]
                clone = getattr(original, "evaluation_copy", None)
                policy = clone() if callable(clone) else copy.deepcopy(original)
                if policy is original:
                    raise ValueError("La validation exige une copie indépendante de l'agent.")
                if deterministic is not None and hasattr(policy, "deterministic"):
                    policy.deterministic = bool(deterministic)
                candidate["policy"] = {"agent_class": type(policy).__name__,
                    "algorithm": type(policy.model).__name__ if getattr(policy, "model", None) is not None else None,
                    "recurrent": bool(getattr(policy, "recurrent", False)),
                    "deterministic": getattr(policy, "deterministic", None)}
                for reference in report["protocol"]["opponents"]:
                    hand_records = []
                    policy_decisions = []
                    rewards, wins, ties, truncated = [], 0, 0, 0
                    actions, positions = [0] * 8, [0] * count
                    # Identical deal schedule across references and training runs.
                    rng = np.random.default_rng(seed)
                    if reference in frozen_opponents:
                        from poker_rl.league import load_frozen
                        reference_agents = [load_frozen(frozen_opponents[reference]) for _ in range(count)]
                    else:
                        reference_agents = [ReferenceAgent(reference, seed + seat + 1) for seat in range(count)]
                    game = PokerGame(**settings)
                    report["protocol"].update(environment={
                        "num_players": count, "initial_stacks": game.table.initial_stacks,
                        "small_blind": game.table.small_blind, "big_blind": game.table.big_blind,
                        "max_actions": game.max_actions,
                        "max_raises_per_player_round": game.table.max_raises_per_player_round},
                        observation_version=game.observation_version)
                    interrupted = False
                    try:
                        for hand in range(hands_per_opponent):
                            if stop_event is not None and stop_event.is_set():
                                interrupted = True
                                break
                            seat = hand % count
                            players = list(reference_agents)
                            players[seat] = policy
                            hand_seed = int(rng.integers(2**63))
                            start = getattr(policy, "on_episode_start", None)
                            if callable(start):
                                start(seed=hand_seed, training=False)
                            game.reset(seed=hand_seed, options={"dealer_seat": (hand // count) % count})
                            game._analysis_preflop_raises = 0
                            for other_seat, other in enumerate(players):
                                if other_seat != seat and callable(getattr(other, 'on_episode_start', None)):
                                    other.on_episode_start(seed=hand_seed + other_seat + 1, training=False)
                            hand_actions = []
                            hand_policy_decisions = []
                            while not game.finished:
                                if stop_event is not None and stop_event.is_set():
                                    interrupted = True
                                    break
                                actor = game.current_seat
                                mask = game.action_masks(actor)
                                action = players[actor].act(game.observe(actor), mask, training=False)
                                action = int(getattr(action, "value", action))
                                if not game.action_space.contains(action) or not mask[action]:
                                    raise ValueError(f"Action illégale en validation : {action}.")
                                if actor == seat:
                                    actions[action] += 1
                                if record_hands or actor == seat:
                                    snapshot = decision_snapshot(game, actor, action)
                                    if actor == seat and getattr(policy, 'last_decision', None):
                                        snapshot['policy'] = copy.deepcopy(policy.last_decision)
                                        hand_policy_decisions.append(snapshot)
                                if record_hands:
                                    hand_actions.append(snapshot)
                                    if snapshot['stage'] == 'PREFLOP' and snapshot['is_raise']:
                                        game._analysis_preflop_raises += 1
                                game.step(action)
                            if interrupted:
                                break
                            positions[seat] += 1
                            if record_hands:
                                hand_records.append(describe_hand(game, seat, hand=hand + 1,
                                    seed=hand_seed, actions=hand_actions))
                            if game.terminated:
                                policy_decisions.extend(hand_policy_decisions)
                                rewards.append(float(game.rewards()[seat]))
                                winners = game.table.winner_indices
                                wins += int(seat in winners and len(winners) == 1)
                                ties += int(seat in winners and len(winners) > 1)
                            else:
                                truncated += 1
                            end = getattr(policy, "on_episode_end", None)
                            if callable(end):
                                end(reward=float(game.rewards()[seat]), terminated=game.terminated,
                                    truncated=game.truncated, training=False)
                            if on_progress and ((hand + 1) % 25 == 0 or hand + 1 == hands_per_opponent):
                                on_progress(trained_seat, reference, hand + 1, hands_per_opponent)
                    finally:
                        game.close()
                    candidate["benchmarks"].append({"opponent": reference,
                        **_summary(rewards, wins, ties, truncated, actions, positions),
                        'policy_diagnostics': summarize_decisions(policy_decisions),
                        **({"hand_analysis": analyze_hands(hand_records)} if record_hands else {})})
                    if interrupted:
                        candidate["status"] = report["status"] = "interrupted"
                        break
            except Exception as exc:
                candidate.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                report["status"] = "failed"
            if stop_event is not None and stop_event.is_set():
                report["status"] = "interrupted"
                break
    return report


def evaluate_depths(agents, training_seats, *, stack_depths=(20, 50, 100), game_kwargs=None, **kwargs):
    """Same frozen policies and deal schedule at each depth, separate scenarios."""
    depths = tuple(stack_depths)
    if not depths or len(set(depths)) != len(depths) or any(d <= 0 for d in depths):
        raise ValueError('Profondeurs positives et distinctes requises.')
    combined = None
    settings = dict(game_kwargs or {})
    for depth in depths:
        settings['initial_stacks'] = depth * settings.get('big_blind', 2)
        report = evaluate_agents(agents, training_seats, game_kwargs=settings, **kwargs)
        for candidate in report['agents']:
            for benchmark in candidate['benchmarks']:
                benchmark.update(depth_bb=depth, scenario=f"{depth}bb:{benchmark['opponent']}")
        if combined is None:
            combined = report
            combined['protocol'].update(stack_depths=list(depths), scenario_weighting='equal',
                frozen_opponents={key: {k: v for k, v in spec.items() if k != 'path'}
                                  for key, spec in kwargs.get('frozen_opponents', {}).items()})
        else:
            for target, candidate in zip(combined['agents'], report['agents']):
                target['benchmarks'].extend(candidate['benchmarks'])
                if candidate['status'] != 'completed':
                    target.update(status=candidate['status'], error=candidate.get('error', ''))
            if report['status'] != 'completed':
                combined['status'] = report['status']
        if report['status'] != 'completed':
            break
    return combined


def attach_validation(recorder, report):
    """Keep validation apart from training statistics, bound to this checkpoint."""
    if recorder is None:
        return
    for candidate in report["agents"]:
        recorder.manifest["agents"][candidate["seat"]]["validation"] = {
            **compact_candidate(candidate, hands_file="validation.json"),
            "protocol": report["protocol"], "created_at": report["created_at"]}
    write_json(recorder.path / "validation.json", report)
    recorder.flush()


def validate_model(registry, model_id, *, hands_per_opponent=DEFAULT_VALIDATION_HANDS,
                   stop_event=None, on_progress=None, stack_depths=(20, 50, 100)):
    """Backfill held-out metrics for an existing checkpoint without retraining it."""
    import json
    model = registry.resolve_model(model_id)
    if model is None:
        raise ValueError("Modèle introuvable.")
    manifest_path = model["run_path"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class SavedPolicy:
        def evaluation_copy(self):
            return registry.load(model_id)
    agents = [None] * model["num_players"]
    agents[model["seat"]] = SavedPolicy()
    report = evaluate_depths(agents, (model["seat"],), game_kwargs=manifest["environment"],
        hands_per_opponent=hands_per_opponent, stop_event=stop_event, on_progress=on_progress,
        stack_depths=stack_depths, opponents=('random', 'check_call', 'tight', 'aggressive'))
    candidate = report["agents"][0]
    filename = f"validation_seat_{model['seat']}.json"
    validation = {**compact_candidate(candidate, hands_file=filename),
                  "protocol": report["protocol"], "created_at": report["created_at"]}
    manifest["agents"][model["seat"]]["validation"] = validation
    # Each seat also has a separate report if an archive contains multiple learners.
    write_json(model["run_path"] / filename, report)
    write_json(manifest_path, manifest)
    return validation
