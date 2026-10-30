import time
from collections import Counter

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from poker_rl.table_session import TableSession


class TableTrainingEnv(gym.Env):
    def __init__(self, session, seat, *, normalize_rewards=True, augment_suits=True, stack_depths=None,
                 curriculum_probability=0., curriculum_warmup=1024, postflop_target=.35):  # Prépare l'adaptateur
        super().__init__()
        # SB3 seeds a newly created model's VecEnv, but set_env() on a resumed
        # model does not. Seed here as well so suit/stack augmentation never
        # falls back to OS entropy when continuing in a fresh table session.
        super().reset(seed=session.seed)
        self.session = session
        self.seat = seat
        self.observation_space = session.game.observation_space
        self.action_space = session.game.action_space
        self.augment_suits = augment_suits
        self.normalize_rewards = normalize_rewards
        self.stack_depths = tuple(stack_depths or ())
        self._hand = None
        self._suits = None
        self.curriculum_rng = np.random.default_rng(np.random.SeedSequence([session.seed, 74113]))
        self.curriculum_probability = curriculum_probability
        self.curriculum_warmup, self.postflop_target = curriculum_warmup, postflop_target
        self.decisions_by_street = Counter()
        self.natural_by_street, self.curriculum_by_street = Counter(), Counter()
        self.prefix_actions = Counter()
        self.curriculum_episodes = Counter()
        self.origin = 'natural'
        self._prepared_hand = None
        session.training_coverage = self.coverage()
 
    @property
    def reward_scale(self):  # Calcule l'échelle des récompenses
        table = self.session.game.table
        if self.normalize_rewards:  # Exprime le gain en fraction du stack
            return table.big_blind / table.initial_stacks
        return 1.0

    def coverage(self):  # Résume les situations rencontrées
        total = sum(self.decisions_by_street.values())
        postflop = total - self.decisions_by_street.get('PREFLOP', 0)
        postflop_fraction = None
        if total > 0:  # Évite une division par zéro
            postflop_fraction = postflop / total
        return {'data_role': 'training_diagnostic', 'decisions_by_street': dict(self.decisions_by_street),
                'natural_decisions_by_street': dict(self.natural_by_street),
                'curriculum_decisions_by_street': dict(self.curriculum_by_street),
                'prefix_actions_by_street': dict(self.prefix_actions),
                'curriculum_episodes': dict(self.curriculum_episodes),
                'postflop_fraction': postflop_fraction,
                'curriculum_probability': self.curriculum_probability,
                'warmup_decisions': self.curriculum_warmup, 'postflop_target': self.postflop_target}

    def _prepare_hand(self):  # Prépare un exercice postflop
        if self._prepared_hand == self.session.hand:  # Évite de préparer deux fois la même main
            return
        self._prepared_hand = self.session.hand
        self.origin = 'natural'
        total = sum(self.decisions_by_street.values())
        natural_total = sum(self.natural_by_street.values())
        natural_postflop = natural_total - self.natural_by_street.get('PREFLOP', 0)
        insufficient = not natural_total or natural_postflop / natural_total < self.postflop_target
        late_decisions = self.natural_by_street.get('TURN', 0) + self.natural_by_street.get('RIVER', 0)
        rare_late_streets = late_decisions < .1 * max(1, natural_total)
        if total < self.curriculum_warmup:  # Attend assez de décisions avant le curriculum
            return
        if not (insufficient or rare_late_streets):  # Garde les mains naturelles si la couverture suffit
            return
        if self.curriculum_rng.random() >= self.curriculum_probability:  # Limite la fréquence des exercices postflop
            return
        streets = ('FLOP', 'TURN', 'RIVER')
        weights = []
        for street in streets:
            weights.append(1 / (1 + self.curriculum_episodes.get(street, 0)))
        target = str(self.curriculum_rng.choice(streets, p=np.asarray(weights) / sum(weights)))
        session = self.session
        session.curriculum_prefix = True
        try:
            while not session.game.finished and session.game.table.stage.name != target:
                if session.stop_event.is_set():  # Interrompt la simulation si l'arrêt est demandé
                    raise InterruptedError('Entraînement arrêté.')
                actor = session.game.current_seat
                session.choose_action()
                mask = session.game.action_masks(actor)
                action = None
                for candidate in (1, 2, 0):
                    if mask[candidate]:  # Choisit un coup autorisé
                        action = candidate
                        break
                if action is None:  # Refuse un préfixe sans coup légal
                    raise ValueError("Aucun check, call ou fold disponible.")
                self.prefix_actions[session.game.table.stage.name] += 1
                session.play(action)
            if not session.game.finished:  # Compte l'exercice si la street ciblée est atteinte
                self.origin = target
                self.curriculum_episodes[target] += 1
        finally:
            session.curriculum_prefix = False
        session.training_coverage = self.coverage()

    def _observation(self):  # Récupère les données du réseau
        observation = self.session.game.observe(self.seat)
        if self.augment_suits:  # Active la permutation des couleurs des cartes
            if self._hand != self.session.hand:  # Change la permutation seulement entre les mains
                self._hand = self.session.hand
                self._suits = self.np_random.permutation(4)
            observation[:104] = observation[:104].reshape(2, 13, 4)[..., self._suits].reshape(-1)
        return observation

    def action_masks(self):  # Renvoie les actions légales
        return self.session.game.action_masks(self.seat)

    def _opponents(self):  # Joue les tours adverses
        session = self.session
        while not session.game.finished and session.game.current_seat != self.seat:
            if session.stop_event.is_set():  # Interrompt la simulation si l'arrêt est demandé
                raise InterruptedError("Entraînement arrêté.")
            session.play(session.choose_action())
        if session.game.finished:  # Enregistre le résultat de la main terminée
            session.finish_hand()

    def reset(self, *, seed=None, options=None):  # Prépare une nouvelle main
        super().reset(seed=seed)
        while True:
            if self.session.stop_event.is_set():  # Respecte la demande d'arrêt
                raise InterruptedError("Entraînement arrêté.")
            if self.session.game.finished:  # Termine la main précédente avant la suivante
                self.session.finish_hand()
                if self.stack_depths:  # Tire une nouvelle profondeur de stack
                    self.session.game.table.initial_stacks = float(self.np_random.choice(self.stack_depths)) * self.session.game.table.big_blind
                self.session.advance()
            self._prepare_hand()
            self._opponents()
            if not self.session.game.finished:  # Renvoie une situation où le bot peut jouer
                return self._observation(), self.session.game.info(self.seat)

    def step(self, action):  # Joue une décision du bot
        session = self.session
        street = session.game.table.stage.name
        self.decisions_by_street[street] += 1
        if self.origin == 'natural':  # Sépare les décisions naturelles des exercices
            self.natural_by_street[street] += 1
        else:
            self.curriculum_by_street[street] += 1
        session.training_coverage = self.coverage()
        session._deliver(self.seat)
        session.play(int(action))
        self._opponents()
        game = session.game
        return (self._observation(), float(game.rewards()[self.seat]) * self.reward_scale,
                game.terminated, game.truncated, game.info(self.seat))


class _Progress(BaseCallback):
    def __init__(self, session, start, total, callback, on_rollout=None):  # Prépare l'adaptateur
        super().__init__()
        self.session, self.start, self.total, self.callback = session, start, total, callback
        self.rollout_callback = on_rollout

    def _on_rollout_start(self):  # Signale le début d'une collecte
        if self.rollout_callback:  # Signale une nouvelle collecte si demandé
            self.rollout_callback(self.session, self.num_timesteps - self.start)

    def _on_step(self):  # Gère la progression et l'arrêt
        while self.session.pause_event.is_set() and not self.session.stop_event.is_set():
            time.sleep(0.02)
        if self.callback and self.n_calls % 32 == 0:  # Actualise le suivi toutes les 32 décisions
            self.callback(self.num_timesteps - self.start, self.total, self.session.stats)
        return not self.session.stop_event.is_set()


def train_sb3_session(registry, selections, *, training_seats, total_timesteps=10000,
                      seed=42, game_kwargs=None, max_hands_per_match=1000,
                      record=True, stop_event=None, on_progress=None, agents=None,
                      validation_hands=500, on_validation_progress=None,
                      independent_hands=True, normalize_rewards=None, augment_suits=True,
                      stack_depths=None, on_session=None, on_rollout=None, perform_validation=True,
                      curriculum_probability=0., curriculum_warmup=1024, postflop_target=.35):  # Lance PPO puis la validation
    if len(training_seats) != 1:  # Un seul siège apprend à la fois
        raise ValueError("L'entraînement SB3 contrôle un seul siège à la fois ; choisissez un seul agent à entraîner.")
    seat = next(iter(training_seats))
    if type(total_timesteps) is not int or total_timesteps < 2:  # Vérifie le budget de décisions
        raise ValueError("Choisissez au moins 2 décisions PPO.")
    if stack_depths:  # Vérifie le mode et les profondeurs de stack
        if not independent_hands:  # Exige la remise à zéro entre les mains
            raise ValueError('Les stacks variables nécessitent des mains indépendantes.')
        for depth in stack_depths:
            if depth <= 0:  # Refuse un stack nul ou négatif
                raise ValueError('Les profondeurs de stack doivent être positives.')
    if not 0 <= curriculum_probability <= 1 or curriculum_warmup < 0 or not 0 <= postflop_target <= 1:  # Vérifie les réglages du curriculum
        raise ValueError('Paramètres de curriculum invalides.')
    if curriculum_probability and not independent_hands:  # Réserve le curriculum aux mains indépendantes
        raise ValueError('Le curriculum nécessite des mains indépendantes.')
    if stack_depths:  # Choisit le stack de la première main
        game_kwargs = dict(game_kwargs or {})
        game_kwargs['initial_stacks'] = float(np.random.default_rng(seed).choice(stack_depths)) * game_kwargs.get('big_blind', 2)
    torch.set_num_threads(1)
    session = TableSession(registry, selections, mode="train", training_seats=training_seats,
        seed=seed, game_kwargs=game_kwargs, continuous=not independent_hands, defer_callbacks=True,
        max_hands_per_match=max_hands_per_match, record=record, agents=agents)  # Crée la table d'entraînement
    if stop_event is not None:  # Partage le signal d'arrêt du lanceur
        session.stop_event = stop_event
    agent = session.agents[seat]
    if normalize_rewards is None:  # Conserve les unités du critique lors d'une reprise
        normalize_rewards = agent.model is None or getattr(agent.model, 'poker_reward_scale', 1.0) != 1.0
    if agent.model is None and agent.seed is None:  # Fixe la seed d'un nouveau modèle
        agent.seed = seed
    if on_session:  # Signale la création de la session
        on_session(session)
    if session.recorder:  # Archive le protocole pour reproduire l'expérience
        reward_unit = 'bb'
        if normalize_rewards:  # Archive l'unité utilisée par le critique
            reward_unit = 'initial_stack'
        session.recorder.manifest['training_protocol'] = {
            'independent_hands': independent_hands, 'normalize_rewards': normalize_rewards,
            'augment_suits': augment_suits, 'reward_unit': reward_unit,
            'curriculum': {'probability': curriculum_probability, 'warmup_decisions': curriculum_warmup,
                           'postflop_target': postflop_target, 'prefix_policy': 'legal_check_call'}}
        session.recorder.flush()
    try:
        if not getattr(agent, "native_training", False):  # Vérifie que l'agent peut s'entraîner avec SB3
            raise TypeError("Cet agent ne fournit pas l'entraînement natif SB3.")
        if session.has_human:  # Exige des adversaires automatiques pendant l'apprentissage
            raise ValueError("L'entraînement SB3 nécessite des adversaires automatiques.")
        env = TableTrainingEnv(session, seat,
                               normalize_rewards=normalize_rewards, augment_suits=augment_suits, stack_depths=stack_depths,
                               curriculum_probability=curriculum_probability, curriculum_warmup=curriculum_warmup,
                               postflop_target=postflop_target)  # Adapte la table à Gymnasium
        start = agent.training_stats["timesteps"]
        callback = _Progress(session, start, total_timesteps, on_progress, on_rollout)  # Relie PPO au suivi
        try:
            agent.learn(env, total_timesteps=total_timesteps, callback=callback)  # Lance l'apprentissage SB3
        except InterruptedError:
            if not session.stop_event.is_set():  # Propage une interruption qui n'a pas été demandée
                raise
        if not session.stop_event.is_set():  # Termine la main visible sauf en cas d'arrêt
            while not session.game.finished:
                session.play(session.choose_action())
            session.finish_hand()
        session.sb3_timesteps = agent.training_stats["timesteps"] - start  # Compte les décisions de cette session
        session.sb3_requested_steps = total_timesteps
        if perform_validation:  # Évalue le modèle sans modifier ses poids
            session.complete_training(validation_hands=validation_hands,
                                      on_validation_progress=on_validation_progress)
        if on_progress:  # Envoie le bilan final au suivi de progression
            on_progress(session.sb3_timesteps, total_timesteps, session.stats)
        return session
    except BaseException as exc:  # Conserve les traces même en cas d'échec
        session.finish_recording(status="failed", error=exc)
        session.close()
        raise
