# neuron_poker — environment minimal

Version minimale dérivée de `dickreuter/neuron_poker` contenant uniquement :

- l'environnement Gym Texas Hold'em ;
- les règles / gestion des tours ;
- le rendu Pyglet ;
- l'évaluation des mains ;
- un Monte Carlo utilisé uniquement pour calculer l'equity affichée.

## Aucun agent / aucune IA

Il n'y a pas de dossier `agents/`, pas de DQN, pas de réseau de neurones, pas de RL et aucune politique qui choisit les actions automatiquement.
Les actions doivent être fournies de l'extérieur via `env.step(...)`.

Le Monte Carlo ne sélectionne jamais une action : il estime seulement la probabilité de victoire affichée comme `EQ: xx%`.

## Installation

Version vérifiée sous Windows avec **Python 3.12**. Les dépendances gardent NumPy
en version 1.x et Pyglet en version 1.5.31 pour ce rendu OpenGL.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Lancement

Sous Windows, double-cliquer sur **`start.bat`** : il prépare l'environnement
si nécessaire, vérifie les dépendances et ouvre la fenêtre de poker.

Ou lancer depuis PowerShell dans ce dossier :

```powershell
.\.venv\Scripts\python.exe example.py
```

La fenêtre reste ouverte jusqu'à sa fermeture. Tu contrôles **les deux joueurs**,
chacun à son tour. Seules les actions disponibles sont affichées dans la fenêtre.

| Touche | Action |
| --- | --- |
| `F` | Se coucher |
| `Espace` | Parole / check |
| `C` | Suivre / call |
| `B` | Relancer à 3 grosses blindes |
| `H` | Relancer d'un demi-pot |
| `P` | Relancer d'un pot |
| `D` | Relancer de deux pots |
| `A` | Tapis / all-in |
| `R` | Recommencer une partie |
| `Échap` | Fermer la fenêtre |

Sur Linux/macOS, utiliser `python3 -m venv .venv`, puis
`.venv/bin/python -m pip install -r requirements.txt` et
`.venv/bin/python example.py`.

## Utilisation dans un programme

L'environnement conserve l'API historique : `reset()` renvoie l'observation et
`step()` renvoie `(observation, reward, done, info)`. Désactiver le vérificateur
de l'API moderne de Gym comme dans l'exemple :

```python
from types import SimpleNamespace
import gym
import gym_env
from gym_env.enums import Action

env = gym.make("neuron_poker-v0", render=False, disable_env_checker=True)
env.add_player(SimpleNamespace(name="Joueur 1"))
env.add_player(SimpleNamespace(name="Joueur 2"))
observation = env.reset()
print(env.legal_moves)
observation, reward, done, info = env.step(Action.CALL.value)
env.close()
```

## Vérification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Structure

```text
gym_env/
  __init__.py
  env.py
  cycle.py
  enums.py
  rendering.py
tools/
  __init__.py
  helper.py
  hand_evaluator.py
  montecarlo_python.py
```

Source originale : `dickreuter/neuron_poker` (licence MIT).
