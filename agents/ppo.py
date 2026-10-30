from pathlib import Path
import tempfile

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from poker_rl.features import PokerFeaturesExtractor
from poker_rl.policy_diagnostics import decision_diagnostics


class PPOAgent:
    MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ppo_poker"
    native_training = True

    def __init__(self, seed=None, checkpoint=None, *, autoload=True, deterministic=None,
                 poker_features=True, **ppo_kwargs):  # Initialise l'agent
        self.seed = seed
        self.model = None
        self.rng = np.random.default_rng(seed)
        self._deterministic_override = deterministic
        self.deterministic = bool(deterministic)
        self.poker_features = poker_features
        self.ppo_kwargs = dict(n_steps=128, batch_size=64, gae_lambda=0.95)
        self.ppo_kwargs.update(ppo_kwargs)
        self.on_episode_start() # Préparation d'une nouvelle main
        default_file = self.MODEL_PATH.with_suffix(".zip")
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        elif autoload and default_file.exists():
            self.load_checkpoint(default_file)

    def act(self, observation, action_mask, *, training=False):  # Choisit une action légale
        probabilities = self.action_probabilities(observation, action_mask)
        deterministic = self.deterministic and not training
        if deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self.rng.choice(len(probabilities), p=probabilities))
        self.last_decision = decision_diagnostics(probabilities, action, deterministic=deterministic) # Analyse le coup joué 
        # permet d'analyser la policy par la suite
        return action

    def action_probabilities(self, observation, action_mask):  # Calcule les probabilités (inférence)
        if self.model is None:
            raise RuntimeError("Entraînez ou chargez un modèle PPO d'abord.")
        mask = np.asarray(action_mask, dtype=bool).flatten()
        if mask.size != self.model.action_space.n or not mask.any():
            raise ValueError("Le masque doit contenir les actions légales du modèle.")
        policy = self.model.policy
        policy.set_training_mode(False)
        obs = np.asarray(observation, dtype=np.float32).flatten()
        tensor, _ = policy.obs_to_tensor(obs)
        with torch.no_grad(): # inférence -> on ne calcule pas le grad
            distribution = policy.get_distribution(tensor, action_masks=mask) # on masque les coups illégaux
            probabilities = distribution.distribution.probs.cpu().numpy().flatten().astype(np.float64)
        return probabilities / probabilities.sum()

    def on_episode_start(self, *, seed=None, training=False):  # Prépare une nouvelle main
        self.last_decision = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def observe(self, transition):  # Collecte gérée par SB3 / interface par défaut (train adapté pour SB3)
        pass

    # === Monitoring du training ===

    @property
    def training_stats(self):  # Lit les statistiques PPO (juste pour monitorer)
        timesteps = 0
        if self.model is not None:  # Lit le compteur du modèle entraîné
            timesteps = self.model.num_timesteps
        stats = {"timesteps": timesteps,
                 "ppo_epochs": getattr(self.model, "_n_updates", 0)}
        logger = getattr(self.model, "_logger", None)
        if logger:
            for name, value in logger.name_to_value.items():
                if name.startswith("train/") and np.isscalar(value) and np.isfinite(value):
                    stats[name.removeprefix("train/")] = float(value)
        return stats

    # === Entrainement (géré par SB3) ===

    def learn(self, env, total_timesteps=200_000, callback=None):  # Entraîne PPO avec GAE
        if type(total_timesteps) is not int or total_timesteps < 2:  # Refuse un budget invalide
            raise ValueError("PPO nécessite au moins 2 décisions d'entraînement.")
        if self.model is None:  # Crée un modèle au premier entraînement
            options = self.ppo_kwargs.copy()
            options["n_steps"] = min(options["n_steps"], total_timesteps)  # Plafonne la collecte au budget (éviter d'être bloqué)
            options["batch_size"] = min(options["batch_size"], options["n_steps"])  # Adapte les lots à la collecte
            policy_kwargs = {}
            if self.poker_features:  # Ajoute les caractéristiques poker aux observations
                policy_kwargs["features_extractor_class"] = PokerFeaturesExtractor # on donne les mains que l'on a (pas sa valeur)
            policy_kwargs.update(options.pop("policy_kwargs", {}))
            self.model = MaskablePPO("MlpPolicy", env, seed=self.seed, device="cpu",
                                     policy_kwargs=policy_kwargs, **options)
        else:  # Reprend les poids et l'optimiseur existants
            self.model.set_env(env)
        self.model.poker_reward_scale = getattr(env, "reward_scale", 1.0)  # Mémorise l'unité des récompenses
        self.model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)  # Collecte, GAE et mises à jour PPO
        self.on_episode_start()  # Prépare le bot à rejouer
        return self.model

    # === On enregistre les modèles (checkpoint) ===

    def save_checkpoint(self, path):  # Sauvegarde le modèle
        if self.model is None:
            raise ValueError("Aucun modèle PPO à sauvegarder.")
        self.model.poker_deterministic = self.deterministic
        self.model.poker_rng_state = self.rng.bit_generator.state
        with Path(path).open("wb") as stream:
            self.model.save(stream)

    def load_checkpoint(self, path):  # Recharge le modèle
        with Path(path).open("rb") as stream:
            try:
                model = MaskablePPO.load(stream, device="cpu")
            except Exception as error:
                raise ValueError("Checkpoint incompatible : créez un nouveau modèle PPO sans mémoire.") from error
        if not isinstance(model.policy, MaskableActorCriticPolicy):
            raise ValueError("Checkpoint incompatible : créez un nouveau modèle PPO sans mémoire.")
        self.model = model
        self.poker_features = isinstance(model.policy.features_extractor, PokerFeaturesExtractor)
        if self._deterministic_override is None:
            self.deterministic = getattr(model, "poker_deterministic", self.deterministic)
        else:
            self.deterministic = bool(self._deterministic_override)
        if hasattr(model, "poker_rng_state"):
            self.rng.bit_generator.state = model.poker_rng_state
        if self.seed is not None:
            model.set_random_seed(self.seed)
        self.on_episode_start()

    def evaluation_copy(self):  # Copie le modèle pour l'évaluation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.checkpoint"
            self.save_checkpoint(path)
            return type(self)(checkpoint=path, autoload=False, deterministic=self.deterministic)

    @classmethod
    def train(cls, env, total_timesteps=200_000, seed=None):  # Entraîne et sauvegarde
        agent = cls(seed=seed, autoload=False)  # Part de poids neufs
        model = agent.learn(env, total_timesteps=total_timesteps)  # Apprend en jouant
        cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier de sauvegarde
        agent.save_checkpoint(cls.MODEL_PATH.with_suffix(".zip"))  # Sauvegarde poids et optimiseur
        return model
