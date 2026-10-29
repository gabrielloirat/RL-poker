"""Evaluate frozen agents in complete matches, retaining stacks between hands."""

from poker_rl.records import write_json
from poker_rl.table_session import TableSession


def run_arena(registry, selections, *, matches=100, seed=42, game_kwargs=None,
              max_hands_per_match=1000, records_dir=None, on_progress=None, stop_event=None):
    if type(matches) is not int or matches < 1:
        raise ValueError("Le nombre de parties d'arène doit être un entier positif.")
    session = TableSession(registry, selections, mode="infer", continuous=True, seed=seed,
                           game_kwargs=game_kwargs, max_hands_per_match=max_hands_per_match,
                           defer_callbacks=True, record=True, records_dir=records_dir, record_kind="arena")
    previous = -1
    try:
        if session.has_human:
            raise ValueError("L'arène nécessite uniquement des agents automatiques.")
        while session.stats.matches < matches:
            if stop_event is not None and stop_event.is_set():
                break
            session.run_batch(matches)
            if session.stats.matches != previous and on_progress:
                on_progress(session.stats)
            previous = session.stats.matches
        stats = session.stats
        report = stats.to_dict()
        report["requested_matches"] = matches
        report["players"] = []
        for seat, (name, _) in enumerate(selections):
            report["players"].append({"seat": seat, "name": name, "wins": stats.wins[seat],
                "losses": stats.matches_completed - stats.wins[seat],
                "win_rate": stats.wins[seat] / stats.matches_completed if stats.matches_completed else None,
                "hand_wins": stats.hand_wins[seat], "hand_ties": stats.hand_ties[seat],
                "net_bb": stats.net_bb[seat], "bb_per_hand": stats.mean_bb[seat]})
        status = "completed" if stats.matches >= matches else "interrupted"
        path = session.finish_recording(status=status)
        report.update(status=status, archive=str(path))
        write_json(path / "arena.json", report)
        return report
    except BaseException as exc:
        session.finish_recording(status="failed", error=exc)
        raise
    finally:
        session.close()
