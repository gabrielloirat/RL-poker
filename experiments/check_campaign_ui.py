"""Render the app's actual monitoring screens for visual regression checks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyglet
from poker_rl.gui import PokerApp

out = Path('runs/ui_checks')
out.mkdir(parents=True, exist_ok=True)
app = PokerApp()
app.window.set_visible(False)
app.campaign.open()


def capture(name):
    app.on_draw()
    pyglet.image.get_buffer_manager().get_color_buffer().save(str(out / f'campaign_{name}.png'))
    print(name, len(app.buttons), flush=True)


try:
    for tab in ('config', 'train', 'validation', 'modes', 'seeds', 'depths', 'league'):
        app.campaign.tab = tab
        capture(tab)
    data = app.campaign.data
    complete = [r for r in (data or {}).get('runs', []) if r.get('final_validation')]
    if complete:
        if data.get('trials'):
            app.campaign.trial_index = next(i for i, t in enumerate(data['trials']) if t['id'] == complete[0]['trial_id'])
        app.campaign.seed_index = app.campaign.visible_runs().index(complete[0])
        app.campaign.inspect()
    else:
        model = next((key for key, value in app.registry.saved_models.items()
                      if key.startswith('current_') and value['num_players'] == 2 and value.get('validation')), None)
        if model:
            app.open_model_metrics(model)
    if app.metrics_open:
        capture('metrics')
        app.open_hand_analysis()
        # Read background traces without pumping live table gameplay.
        if app.pending is not None:
            app.pending.result(timeout=60)
            app.tick(0)
        for view in ('chart', 'ranges', 'style', 'policy'):
            app.analysis_view = view
            capture(view)
        app.analysis_parent, app.analysis_dimension = 'ranges', 'starting_hand'
        holding = next((h['starting_hand'] for h in app.analysis_hands or [] if h.get('starting_hand')), None)
        app.show_analysis_hands(category=holding, outcome='all')
        capture('range_hands')
        if app.analysis_hands:
            app.inspect_analysis_hand(app.analysis_hands[0])
            capture('hand_detail')
    print('UI checks completed', flush=True)
finally:
    app.on_close()
    app.viewer.close()
