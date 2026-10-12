# Déposez vos agents ici

## Exemple fourni : agent humain

`human.py` est un plugin complet qui joue l'action saisie dans le terminal.
Dans l'interface graphique (`example.py` ou `start.bat`), sélectionnez `human`
pour les sièges que vous voulez contrôler : les actions sont alors saisies via
les boutons de la fenêtre ou les raccourcis clavier. Il fonctionne avec 2 à 5
joueurs. Pour utiliser le terminal avec deux sièges :

```powershell
.\.venv\Scripts\python.exe run_agents.py infer --agent0 human --agent1 human --episodes 1
```

À chaque tour, il affiche le siège, vos cartes, le board, les stacks, le pot
et uniquement les actions disponibles. Les montants sont exprimés en pourcentage
du stack initial pour fonctionner avec toutes les configurations de partie.
Entrez le numéro de l'action (`0` à `7`) ou sa lettre (`f/e/c/b/h/p/d/a`).
Une saisie invalide ou une action indisponible est redemandée. `q` ou `Ctrl+C`
ferme la session. Le gain net est affiché en grosses blindes à la fin de la main.

Vous pouvez ensuite remplacer l'un des `human` par le nom de votre agent pour
jouer contre lui. L'agent humain s'utilise en inférence ; pour entraîner votre
propre agent contre un humain, sélectionnez uniquement le siège de votre agent
avec `--train-seats`.

Dans une configuration JSON, `{"name": "human", "kwargs": {"name": "Alice"}}`
permet de personnaliser le nom affiché.

Dans le mode **Voir jouer**, la GUI appelle `act(..., training=False)`.
Dans le mode **Entraînement**, les sièges cochés reçoivent `training=True` et
leurs transitions via `observe(transition)`. Les autres restent en inférence.
L'agent doit implémenter sa propre mise à jour du modèle dans `observe`.
Les callbacks `on_episode_start` et `on_episode_end` reçoivent également le bon
flag `training`, ce qui permet de réserver les sauvegardes et mises à jour au
mode entraînement. Le bouton **Voir ces agents jouer** conserve les instances
entraînées pour les observer ensuite sans apprendre.
Dans la fenêtre, les parties enchaînent plusieurs mains avec les mêmes stacks
jusqu'à l'élimination de tous les joueurs sauf un. Les transitions et callbacks
restent fournis par main ; leurs récompenses sont les gains nets de cette main.
`transition.info` expose aussi `match`, `hand_in_match`, `match_finished` et
`match_winner`. Un siège éliminé ne prend plus de décisions pour la partie.
À plus de deux joueurs, l'observation possède un schéma étendu : consultez
`poker_rl.observation_fields(num_players)` et le README principal. Si votre
modèle attend exactement 147 entrées, utilisez une table à deux joueurs ou
adaptez ses entrées. Les paramètres JSON par siège permettent de passer votre
checkpoint ou le nombre de joueurs à votre constructeur, s'il les accepte.

## Exemple fourni : agent aléatoire

`random.py` choisit uniformément parmi les actions autorisées à son tour.
Il fonctionne avec 2 à 5 joueurs et dispose de son propre générateur aléatoire.
Il ne s'entraîne pas : utilisez-le en inférence ou comme adversaire d'un agent
que vous entraînez.

Dans la fenêtre, cliquez sur **Actualiser les agents**, puis sélectionnez
**random** avec les flèches du siège souhaité. Plusieurs sièges peuvent utiliser
ce plugin, avec une instance indépendante par siège.
Les paramètres JSON `{"seed": 42}` permettent de rendre sa séquence de choix
reproductible ; sans seed, chaque instance utilise un tirage indépendant.

```powershell
# Jouer contre l'agent aléatoire depuis le terminal
.\.venv\Scripts\python.exe run_agents.py infer --agent0 human --agent1 random --episodes 1

# Faire jouer deux agents aléatoires
.\.venv\Scripts\python.exe run_agents.py infer --agent0 random --agent1 random --episodes 100
```

## Ajouter votre agent

Un fichier `mon_agent.py` devient disponible sous le nom `mon_agent`.
Il suffit d'y définir une seule classe avec la méthode :

```python
class MonAgent:
    def __init__(self, checkpoint=None):
        # Initialiser ici votre modèle / charger votre checkpoint.
        ...

    def act(self, observation, action_mask, *, training=False):
        # Retourner un entier 0..7 autorisé par action_mask.
        # Remplacer par votre politique.
        raise NotImplementedError

    def observe(self, transition):
        # Nécessaire seulement pour entraîner cet agent.
        # Stocker la transition et/ou mettre à jour votre modèle.
        ...
```

Ce code est un gabarit, à compléter avec votre politique.
Aucun héritage, décorateur ou enregistrement manuel n'est nécessaire.

Pour un agent composé de plusieurs fichiers :

```text
agents/
  mon_agent/
    agent.py       # classe avec act() ou fonction create_agent(**kwargs)
    model.py
    checkpoint.pt
```

Dans `agent.py`, utilisez `from .model import ...` pour vos imports locaux.
Pour un checkpoint livré dans le dossier, construisez son chemin avec
`Path(__file__).resolve().parent / "checkpoint.pt"`.
Un paquet avec `__init__.py` peut aussi exposer l'agent directement si aucun
`agent.py` n'est présent. Si plusieurs classes ont `act()`, définissez une
fonction `create_agent(**kwargs)` qui retourne l'instance à utiliser.

Les fichiers/dossiers commençant par `_` ou `.` sont ignorés ; les autres
fichiers `.py` à la racine sont considérés comme des plugins. Placez vos helpers
dans le dossier de leur agent, ou préfixez-les par `_`.
Chaque nom doit être un identifiant Python et être unique : ne créez pas à la
fois `mon_agent.py` et `mon_agent/`.

```powershell
.\.venv\Scripts\python.exe run_agents.py list
.\.venv\Scripts\python.exe run_agents.py infer --agent0 mon_agent --agent1 autre_agent --episodes 100
.\.venv\Scripts\python.exe run_agents.py train --agent0 mon_agent --agent1 autre_agent --train-seats 0
```

Installez les dépendances de vos agents dans le même environnement Python.
La découverte importe les plugins sans construire leur modèle : gardez
l'initialisation du modèle dans `__init__` ou `create_agent`, et protégez vos
scripts d'entraînement par `if __name__ == "__main__":`.
Relancez la commande après avoir ajouté ou modifié un agent.
