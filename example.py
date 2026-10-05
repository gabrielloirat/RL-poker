"""Play both seats manually in the Pyglet window; no built-in agent."""
import gym
import pyglet
from pyglet.window import key

import gym_env  # registers neuron_poker-v0
from gym_env.enums import Action
from gym_env.rendering import WHITE


class Player:
    def __init__(self, name):
        self.name = name


KEY_ACTIONS = {
    key.F: Action.FOLD,
    key.SPACE: Action.CHECK,
    key.C: Action.CALL,
    key.B: Action.RAISE_3BB,
    key.H: Action.RAISE_HALF_POT,
    key.P: Action.RAISE_POT,
    key.D: Action.RAISE_2POT,
    key.A: Action.ALL_IN,
}
ACTION_LABELS = {
    Action.FOLD: "F: se coucher",
    Action.CHECK: "Espace: parole",
    Action.CALL: "C: suivre",
    Action.RAISE_3BB: "B: 3 BB",
    Action.RAISE_HALF_POT: "H: demi-pot",
    Action.RAISE_POT: "P: pot",
    Action.RAISE_2POT: "D: 2 pots",
    Action.ALL_IN: "A: tapis",
}


def main():
    env = gym.make(
        "neuron_poker-v0", initial_stacks=500, render=True,
        funds_plot=False, disable_env_checker=True,
    )
    table = env.unwrapped
    env.add_player(Player("Joueur 1"))
    env.add_player(Player("Joueur 2"))
    try:
        env.reset()
        window = table.viewer.display_surface
        window.set_size(window.width, 600)
        table.viewer.top = 570
        status = "Vous controlez les deux joueurs, chacun a son tour."

        @window.event
        def on_draw():
            table.render()
            viewer = table.viewer
            if table.done:
                heading = f"Partie terminee : {table.players[table.winner_ix].name} gagne."
            else:
                heading = f"A vous : {table.current_player.name} - {table.stage.name}"
            viewer.text(heading, 20, 410, font_size=12, color=WHITE)
            legal = [ACTION_LABELS[action] for action in table.legal_moves]
            viewer.text(" | ".join(legal[:4]), 20, 440, font_size=10, color=WHITE)
            viewer.text(" | ".join(legal[4:]), 20, 462, font_size=10, color=WHITE)
            viewer.text(status, 20, 490, font_size=10, color=WHITE)
            viewer.text("R: nouvelle partie | Echap: quitter", 20, 520, font_size=10, color=WHITE)

        @window.event
        def on_key_press(symbol, modifiers):
            nonlocal status
            if symbol == key.ESCAPE:
                window.close()
                return
            if symbol == key.R:
                env.reset()
                status = "Nouvelle partie. Vous controlez les deux joueurs."
                return
            action = KEY_ACTIONS.get(symbol)
            if action is None:
                return
            if table.done:
                status = "Appuyez sur R pour recommencer."
            elif action not in table.legal_moves:
                status = "Action indisponible. Utilisez une des touches affichees."
            else:
                player_name = table.current_player.name
                env.step(action.value)
                status = f"{player_name} : {ACTION_LABELS[action].split(': ', 1)[1]}"

        print("La fenetre de poker est ouverte. Actions manuelles pour les deux joueurs.")
        print("F: fold, Espace: check, C: call, B/H/P/D: relancer, A: all-in, R: reset, Echap: quitter.")
        pyglet.app.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
