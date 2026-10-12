# RL-poker — environnement et interface pour vos agents

Projet dérivé de `dickreuter/neuron_poker` avec :

- l'environnement Gym Texas Hold'em ;
- les règles / gestion des tours ;
- le rendu Pyglet ;
- l'évaluation des mains ;
- un Monte Carlo utilisé uniquement pour calculer l'equity affichée ;
- une interface Gymnasium pour entraîner ou exécuter vos propres agents ;
- une configuration permettant de choisir les agents et les sièges à entraîner.

## Vos propres politiques

Le projet fournit un agent humain (`agents/human.py`) qui joue selon les saisies
de l'utilisateur et un agent aléatoire (`agents/random.py`) qui choisit parmi
les actions légales. Vos réseaux de neurones et algorithmes d'apprentissage
viennent de votre code. L'environnement
fournit les observations, les actions légales, les récompenses et les transitions.
Vos agents restent responsables de leur apprentissage et de leurs checkpoints.

Le Monte Carlo ne sélectionne jamais une action : il estime l'equity affichée
comme `EQ : ≈ xx%`, en partageant les égalités. Il est désactivé dans le moteur
RL pour éviter le coût des simulations ; la fenêtre graphique le calcule
séparément en arrière-plan, sans modifier les observations ou l'aléatoire des agents.

## Brancher vos agents

Deux interfaces sont disponibles. **`PokerEnv`** est un environnement Gymnasium
pour un agent contrôlé de l'extérieur face à votre adversaire. **`run_agents.py`**
permet de sélectionner deux agents et de distribuer leurs transitions à un ou
deux sièges en entraînement.

L'interface graphique et `PokerGame` prennent en charge **2 à 5 joueurs**.
`PokerEnv` et le lanceur CLI `run_agents.py` conservent leur interface à deux
joueurs. `run_session` accepte de 2 à 5 objets agents. Dans ces interfaces RL,
un épisode correspond à une main indépendante. **La fenêtre joue des parties
complètes** : les stacks sont conservés entre les mains, les sièges à zéro sont
éliminés et la partie se termine quand il reste un seul joueur avec des jetons.

### Ajout plug and play

Un premier exemple est déjà disponible : **`human`**, contrôlé depuis le terminal.
Pour jouer les deux sièges, chacun à son tour :

```powershell
.\.venv\Scripts\python.exe run_agents.py infer --agent0 human --agent1 human --episodes 1
```

Entrez le numéro ou la lettre d'une action affichée ; `q` quitte la session.
L'agent affiche vos cartes, le board et les montants en pourcentage du stack initial.
Remplacez un des `human` par votre agent pour jouer contre lui. Les noms affichés
peuvent être personnalisés via `kwargs.name` dans la configuration JSON.

Déposez vos agents dans **`agents/`** : chaque fichier `mon_agent.py`, ou dossier
`mon_agent/` contenant `agent.py`, devient disponible sous le nom **`mon_agent`**.
Le module doit définir une seule classe avec `act()`, ou une fonction
`create_agent(**kwargs)` qui construit votre agent. Aucun enregistrement manuel
ni héritage n'est nécessaire. Le [guide du dossier agents](agents/README.md)
contient le gabarit et les conventions pour les imports et checkpoints.

```powershell
# Voir les agents disponibles et les éventuelles erreurs de chargement
.\.venv\Scripts\python.exe run_agents.py list

# Utiliser les agents déposés dans agents/mon_agent.py et agents/mon_adversaire.py
.\.venv\Scripts\python.exe run_agents.py infer --agent0 mon_agent --agent1 mon_adversaire --episodes 100
.\.venv\Scripts\python.exe run_agents.py train --agent0 mon_agent --agent1 mon_adversaire --train-seats 0 --episodes 10000
```

Le dossier `agents/` à la racine du projet est utilisé par défaut, même lorsque
Python est lancé depuis un autre répertoire. `--agents-dir chemin` permet de
choisir un autre dossier. Dans le JSON, `"agents_dir": "chemin"` est résolu
relativement au fichier de configuration ; l'option CLI prend priorité.
Relancez la commande après ajout ou modification d'un agent.

La découverte importe les modules sans instancier les agents. Un plugin dont
l'import échoue reste signalé comme indisponible sans empêcher les autres de
fonctionner. Installez ses dépendances dans votre environnement Python et
conservez le chargement de son modèle dans son constructeur ou sa factory.

```python
from poker_rl import AgentRegistry, PokerEnv, load_agent

registry = AgentRegistry()  # registry.agents : catalogue, registry.errors : erreurs
mon_agent = load_agent("mon_agent", {"checkpoint": "models/agent.pt"})
env = PokerEnv(opponent="mon_adversaire")
# Aussi possible : gym.make("poker-rl-v0", opponent="mon_adversaire")
# agents_dir et opponent_kwargs permettent de personnaliser le chargement.
```

### Contrat d'un agent

Votre agent expose :

```python
act(observation, action_mask, *, training=False) -> int
```

- `observation` : tableau `float32` sans `NaN` (147 valeurs à deux joueurs) ;
- `action_mask` : tableau booléen de taille 8, `True` pour les actions légales ;
- `training` : `True` uniquement pour les sièges sélectionnés en entraînement ;
- résultat : entier entre 0 et 7, choisi par votre propre politique.

Pour utiliser le mode entraînement de `run_agents.py`, votre agent expose aussi :

```python
observe(transition) -> None
```

`transition` est une instance de `poker_rl.Transition`. Elle contient
`observation`, `action_mask`, `action`, `reward`, `next_observation`,
`next_action_mask`, `terminated`, `truncated` et `info`. L'observation suivante
correspond au **prochain tour du même joueur**, après les actions adverses,
ou à la fin de la main. Les transitions sont envoyées aux sièges sélectionnés
uniquement. L'agent décide comment les stocker et quand mettre à jour son modèle.

Deux callbacks facultatifs permettent de gérer les états récurrents et les
checkpoints dans votre propre code :

```python
on_episode_start(*, seed, training)
on_episode_end(*, reward, terminated, truncated, training)
```

En inférence, `act` reçoit `training=False` et `observe` n'est jamais appelé.
Les callbacks facultatifs reçoivent également `training=False` ; votre code doit
respecter ce mode. Les protocoles `Agent` et `TrainableAgent` servent au typage :
aucun héritage n'est obligatoire.

### Choisir les agents et le mode

Les noms découverts dans `agents/` sont utilisables avec `--agent0` et `--agent1`.
L'ancien format `module:ClasseOuFactory` reste accepté pour les modules
importables depuis la racine du projet ou présents sur `PYTHONPATH` :

```powershell
# Entraîner le siège 0 face à votre autre agent
.\.venv\Scripts\python.exe run_agents.py train --agent0 mes_agents:MonAgent --agent1 mes_agents:MonAdversaire --train-seats 0 --episodes 10000

# Entraîner les deux agents
.\.venv\Scripts\python.exe run_agents.py train --agent0 mes_agents:AgentA --agent1 mes_agents:AgentB --train-seats 0 1 --episodes 10000

# Inférence uniquement
.\.venv\Scripts\python.exe run_agents.py infer --agent0 mes_agents:AgentA --agent1 mes_agents:AgentB --episodes 100
```

Pour passer les paramètres de vos constructeurs, copiez
`config.agents.example.json` vers `config.agents.json`, remplacez les noms
`mon_agent` et `mon_adversaire` par ceux de vos plugins, puis renseignez leurs `kwargs`.
Par exemple, un argument de checkpoint peut être transmis dans `kwargs` si votre
constructeur le prend en charge. Le fichier exemple est un gabarit : les agents
mentionnés ne sont pas fournis. La clé `"name"` choisit un plugin ; la clé
historique `"class"` accepte aussi bien un nom de plugin que `module:Factory`.

```powershell
.\.venv\Scripts\python.exe run_agents.py train --config config.agents.json
.\.venv\Scripts\python.exe run_agents.py infer --config config.agents.json
.\.venv\Scripts\python.exe run_agents.py infer --config config.agents.json --render --delay 0.5
```

Dans le JSON, `train: true` sélectionne les sièges à entraîner. En mode `infer`,
ces champs sont ignorés et tous les agents restent en inférence. Sans sélection
explicite en mode `train`, le siège 0 est sélectionné. Les statistiques finales
indiquent le gain net moyen en grosses blindes, les mains terminées, les
troncatures et les actions illégales. La fenêtre affiche les deux mains pour le
débogage ; les agents reçoivent chacun leurs seules cartes privées.

### Utiliser directement Gymnasium

```python
from poker_rl import PokerEnv

# mon_adversaire est votre objet, avec la méthode act décrite ci-dessus.
env = PokerEnv(opponent=mon_adversaire, agent_seat=0)
observation, info = env.reset(seed=42)

# Votre boucle d'entraînement ou d'inférence choisit l'action.
action = mon_agent.act(observation, info["action_mask"], training=True)
observation, reward, terminated, truncated, info = env.step(action)
# Répéter jusqu'à terminated or truncated, puis env.reset().
env.close()
```

`PokerEnv.step` joue automatiquement l'adversaire jusqu'au prochain tour du
siège contrôlé, ou à la fin de la main. La récompense et l'observation restent
toujours celles du siège contrôlé. `env.action_masks()` expose aussi le masque.
L'adversaire doit choisir une action légale ; une action illégale de sa part
lève une erreur explicite.

Le siège contrôlé est dealer par défaut dans `PokerEnv`, pour que `reset()`
renvoie toujours un tour où il peut agir. Vous pouvez utiliser
`reset(options={"dealer_seat": 1})` pour sélectionner une autre position,
à condition que l'adversaire ne termine pas la main avant le premier tour de
l'agent. Les sessions `run_agents.py` gèrent les deux positions, y compris les
mains terminées avant qu'un joueur ait pu agir.

L'environnement est aussi enregistré sous `poker-rl-v0` :

```python
import gymnasium as gym
import poker_rl  # enregistre l'environnement

env = gym.make("poker-rl-v0", opponent=mon_adversaire)
```

### Observations, actions et récompenses

L'observation n'inclut aucune carte privée adverse. Les champs sont concaténés
dans cet ordre ; `poker_rl.OBSERVATION_FIELDS` expose le schéma :

| Champ | Taille | Contenu |
| --- | ---: | --- |
| `hole_cards` | 52 | Cartes privées du joueur, indicatrices |
| `board` | 52 | Cartes publiques, indicatrices |
| `stage` | 4 | Préflop, flop, turn, river |
| `seat` | 2 | Siège absolu du joueur |
| `dealer` | 2 | Dealer, ordre joueur puis adversaire |
| `stacks` | 2 | Stacks, ordre joueur puis adversaire |
| `street_bets` | 2 | Mises de la street |
| `hand_contributions` | 2 | Contributions cumulées de la main |
| `pots` | 2 | Pot précédent et pot de la street |
| `to_call` | 1 | Montant à suivre, plafonné au stack |
| `raises_remaining` | 2 | Relances disponibles, normalisées |
| `last_actions` | 16 | Dernière action de chaque joueur sur la street |
| `action_mask` | 8 | Actions légales au tour courant |

Les montants des observations sont divisés par le total initial des jetons.
L'ordre des 52 cartes est `23456789TJQKA` puis `CDHS` pour chaque rang,
également disponible dans `poker_rl.CARD_ORDER`. Le schéma est versionné par
`info["observation_version"]`.

| Entier | Action |
| ---: | --- |
| 0 | Fold |
| 1 | Check |
| 2 | Call |
| 3 | Raise 3 BB |
| 4 | Raise half pot |
| 5 | Raise pot |
| 6 | Raise 2 pots |
| 7 | All-in |

Les blinds sont placées automatiquement. Le moteur conserve ses tailles
discrètes de mises et sa limite configurable de relances par street ; il ne
constitue pas un moteur complet de toutes les règles de tournois no-limit.

La récompense est nulle pendant une main, puis vaut
`(stack_final - stack_initial) / big_blind` à la fin, pour chaque joueur.
Les blinds et les remboursements sont donc inclus ; une égalité partage le pot.
Les gains nets des deux joueurs ont une somme nulle.

À **3, 4 ou 5 joueurs**, le schéma porte la version 2. Les champs par joueur
(`seat`, `dealer`, `stacks`, `street_bets`, `hand_contributions`,
`raises_remaining`) ont une taille égale au nombre de joueurs ; `last_actions`
contient 8 valeurs par joueur. Un champ `active_players` ajouté à la fin indique
les joueurs encore éligibles, y compris ceux à tapis, et exclut les joueurs
couchés. L'ordre relatif est le joueur courant, puis les sièges suivants
circulairement ; `seat` reste un encodage du siège absolu.
Utilisez `poker_rl.observation_fields(num_players)` ou
`game.observation_fields` pour lire ce schéma (164, 179 et 194 valeurs).
Les agents conçus pour un réseau à 147 entrées doivent être adaptés pour jouer
à plus de deux. L'agent `human` fonctionne avec les quatre tailles de table.
Les contributions sont normalisées par `initial_stacks * num_players` ; les
pots secondaires et les égalités sont réglés par niveau de contribution.

Une action illégale du joueur contrôlé donne une pénalité de `-1`, conserve
l'état du jeu et compte dans `max_actions`. La pénalité n'est pas un transfert
de jetons. Une main atteignant `max_actions` sans se terminer est tronquée, avec
une récompense nette de zéro et `truncated=True` ; le pot reste non distribué.
Appelez `reset()` après toute terminaison ou troncature.

Pour du code multi-agent personnalisé, `PokerGame` expose `current_seat`,
`observe(seat)` et `action_masks(seat)`. Son `step` renvoie
`(observation, rewards_par_siège, terminated, truncated, info)` ; c'est une
interface à tours alternés, tandis que `PokerEnv` est l'interface Gymnasium
à perspective fixe.

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

La fenêtre s'ouvre sur le **choix des agents** :

1. Choisissez **2, 3, 4 ou 5 joueurs**. Seuls ces sièges participent.
2. Pour chaque siège, sélectionnez un agent avec les flèches. Les plugins de
   `agents/` apparaissent automatiquement ; **Actualiser les agents** détecte
   les ajouts récents. Un même plugin peut occuper plusieurs sièges, avec une
   instance distincte par siège.
3. Si nécessaire, modifiez les paramètres JSON du constructeur (par exemple
   `{"checkpoint": "models/agent.pt"}`). Choisissez le mode et le nombre de parties,
   puis lancez la session.

### Modes de la fenêtre

**Voir jouer** exécute les agents avec `training=False`, sans appeler `observe`.
La table, les actions et les equities sont visibles. **Parties** désigne le
nombre de parties complètes à jouer. Chaque partie contient plusieurs **mains**,
et chaque main traverse le préflop, le flop, la turn et la river, sauf lorsqu'un
fold ou des tapis la terminent plus tôt.
**Enchaîner les mains** distribue automatiquement la main suivante en conservant
les stacks et en faisant avancer le dealer parmi les joueurs encore en lice.
Lorsqu'un seul joueur possède des jetons, la partie est gagnée : seule la partie
suivante réinitialise les stacks. Les sièges éliminés ne reçoivent plus d'actions
ni de cartes. Le format des observations conserve les sièges initiaux ; leurs
stacks et cartes sont à zéro pour les éliminés.
Le suivi distingue le nombre de parties, les mains jouées, les victoires par
joueur et le gain moyen en BB par main. **Max. mains / partie** permet de limiter
les parties qui s'éternisent (1 000 par défaut) : elles sont comptées comme
interrompues, sans déclarer de gagnant. Désactivez l'enchaînement pour utiliser
manuellement **Main suivante**, puis **Partie suivante** à la fin de la partie.

**Entraînement** permet de cocher **Entraîner** pour les sièges à entraîner.
Ces agents doivent exposer `observe(transition)` et reçoivent `training=True`
dans leurs actions et callbacks. Les adversaires restent en inférence.
Les transitions aboutissent au prochain tour du même joueur ou à la fin de la
main, comme dans le lanceur CLI. `human` et `random` sont des adversaires : ils
n'ont pas de méthode d'apprentissage et ne peuvent pas être cochés.
Le mode utilise l'algorithme de votre agent ; le moteur ne fournit pas de réseau
ni d'optimiseur. Les modèles et checkpoints restent gérés par vos callbacks.

L'unité des transitions et callbacks RL reste **la main** : `terminated`
signale la fin d'une main, même si la partie continue. La récompense correspond
au gain net de cette main, mesuré depuis ses stacks de départ, et non depuis
le début de la partie. `transition.info` ajoute `match`, `hand_in_match`,
`match_finished` et `match_winner` pour suivre la partie complète. Les stacks
et les instances d'agents sont conservés entre ces épisodes.

**Rapide** simule des lots de tours en arrière-plan et affiche la progression,
les gains moyens et le nombre de transitions reçues par siège. Il évite le coût
du rendu des mains et du Monte Carlo. **Passer au mode visuel** continue avec
les mêmes agents, la table et les equities. Un siège humain impose le mode
visuel afin de pouvoir choisir ses actions. Les opérations d'apprentissage et
les callbacks s'exécutent en arrière-plan pour garder la fenêtre réactive.

Après l'entraînement, **Voir ces agents jouer** passe les **mêmes instances**
en inférence pour une partie complète : les paramètres appris en mémoire sont conservés
et aucune transition d'apprentissage n'est envoyée. Les statistiques de cette
phase d'observation repartent de zéro. Revenir au menu puis lancer une nouvelle
session construit de nouvelles instances ; pour retrouver leurs poids après
un redémarrage, utilisez les checkpoints pris en charge par vos agents.

**Pause** suspend les tours suivants, après l'opération en cours.
**Choisir les agents** arrête la session et revient au menu.

Le bouton **Plein écran [F11]** ou la touche **F11** bascule l'affichage.
En plein écran, **Échap** revient au mode fenêtré sans quitter la partie.
La fenêtre est redimensionnable ; l'interface et les zones de clic s'adaptent
à sa taille en conservant leurs proportions.

Les agents automatiques jouent à leur tour. Pour `human`, cliquez sur un bouton
d'action ou utilisez les raccourcis ci-dessous : **aucune saisie dans le
terminal n'est nécessaire**. Le chargement des modèles et leurs décisions
s'exécutent en arrière-plan pour conserver une fenêtre réactive.
Les boutons s'éclairent au survol, s'enfoncent à l'appui et confirment le clic
dans un bandeau. Relâcher la souris hors du bouton annule le clic. Les contrôles
indisponibles sont grisés ; leur survol ou un clic explique pourquoi.
**Main suivante** conserve les stacks et fait tourner le dealer.
**Partie suivante** remet les stacks à leur valeur initiale après la fin d'une
partie. **Choisir les agents** retourne au menu.

Les cartes adverses sont masquées par défaut ; les cartes du siège humain
actif sont affichées. À l'abattage, les mains non couchées sont révélées.
Le bouton **Afficher les cartes** active un affichage de débogage sans modifier les
observations privées envoyées aux agents.

Chaque joueur affiche son **equity** sous son stack : estimation par 1 000
simulations contre des mains adverses aléatoires, mise à jour lorsque le board
ou les joueurs en lice changent. `EQ : calcul…` apparaît pendant le calcul.
Les joueurs couchés ont 0 % ; les joueurs à tapis restent pris en compte.
À l'abattage, le résultat connu remplace l'estimation (les égalités partagent
l'equity). Ces valeurs servent uniquement à l'affichage.

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
| `F11` | Basculer entre plein écran et fenêtre |
| `Échap` | Quitter le plein écran, sinon retour au menu puis fermeture |

Sur Linux/macOS, utiliser `python3 -m venv .venv`, puis
`.venv/bin/python -m pip install -r requirements.txt` et
`.venv/bin/python example.py`.

## API historique et jeu manuel

`gym_env.HoldemTable` conserve l'API historique : `reset()` renvoie l'observation et
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
poker_rl/
  interface.py       # contrats et chargement de vos agents
  plugins.py         # découverte automatique et catalogue des agents
  game.py            # main de 2 à 5 joueurs, observations privées et récompenses
  env.py             # environnement Gymnasium à perspective fixe
  session.py         # distribution des tours et transitions
  gui.py             # choix des joueurs et actions dans la fenêtre
  table_session.py   # session d'inférence avancée par l'interface
agents/              # déposez ici vos fichiers ou dossiers d'agents
run_agents.py
config.agents.example.json
```

Source originale : `dickreuter/neuron_poker` (licence MIT).
