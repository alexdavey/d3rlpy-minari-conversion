""" Benchmark a d3rlpy dataset or a Minari one, using d3rlpy.

Usage: python benchmark.py

See --help for all options.
"""
import pyrallis
from dataclasses import dataclass

import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
from torch.optim.lr_scheduler import CosineAnnealingLR

import minari
import numpy as np

from utils import evaluate_minari_policy, get_MDPDatasetEnv


@dataclass
class BenchmarkConfig:
    dataset_name: str = "pointmaze-umaze-v0"  # name of the Minari dataset
    algorithm: str = "BC"  # offline RL algorithm to use, from d3rlpy
    total_steps: int = int(5e5)  # total number of steps to train for
    seed: int = 0  # seed for Numpy and the Minari environment
    is_minari_dataset: bool = True  # whether a Minari dataset or the original D4RL dataset
    learning_rate: float = 3e-4  # learning rate
    batch_size: int = 256  # batch size
    use_gpu: bool = True  # whether to use a GPU
    n_eval_episodes: int = 100  # number of episodes to evaluate model on validation dataset
    eval_frequency: int = 2000  # number of update steps before evaluating
    max_episode_steps: int = 300
    ref_min_score: float = 0.291 * 10.0 / 3  # reference minimum/random policy score
    ref_max_score: float = 12.0 * 10.0 / 3  # reference maximum/expert policy score
    # checkpoint_interval: int = 10 # number of epochs/evaluations before saving a checkpoint
    # ref_min_score = 23.85
    # ref_max_score = 161.86


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=BenchmarkConfig)

    print(f"Loading dataset...")
    dataset, env = get_MDPDatasetEnv(
        cfg.dataset_name, cfg.is_minari_dataset, cfg.max_episode_steps
    )

    d3rlpy.seed(cfg.seed)

    # The gynasium envs used in Minari are slightly different to those in the
    # original D4RL, so we use a modified function to step through and evaluate
    # the policy
    eval_func = (
        evaluate_minari_policy if cfg.is_minari_dataset else evaluate_on_environment
    )
    evaluate_scorer = eval_func(env, n_trials=cfg.n_eval_episodes)

    # D4RL reports normalized scores, with ref_min being the score of a random policy
    # and ref_max the expected reward of an expert. We replicate this here
    def normalized_scorer(algo, *args):
        score = evaluate_scorer(algo, *args)
        return (
            100 * (score - cfg.ref_min_score) / (cfg.ref_max_score - cfg.ref_min_score)
        )

    # The original implementation of IQL uses reward scaling.
    # Based on https://github.com/takuseno/d3rlpy/blob/master/reproductions/offline/iql.py
    # if cfg.algorithm == "IQL":
    #    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(multiplier=1000.0)
    # else:
    #    reward_scaler = lambda *args: None

    # Setup offline RL algorithm
    if hasattr(d3rlpy.algos, cfg.algorithm):
        alg = getattr(d3rlpy.algos, cfg.algorithm)
    else:
        raise ValueError(f'No algorithm "{cfg.algorithm}" in d3rlpy library.')

    model = alg(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        use_gpu=cfg.use_gpu
        # reward_scaler=reward_scaler
    )

    # The original implementation of IQL uses a learning scheduler.
    # Based on https://github.com/takuseno/d3rlpy/blob/master/reproductions/offline/iql.py
    if cfg.algorithm == "IQL":
        model.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
        scheduler = CosineAnnealingLR(model.impl._actor_optim, cfg.total_steps)
        callback = lambda *args: scheduler.step()
    else:
        callback = lambda *args: None

    # Begin training
    training_results = model.fit(
        dataset,
        eval_episodes=dataset.episodes,
        n_steps=cfg.total_steps,
        n_steps_per_epoch=cfg.eval_frequency,
        scorers={"environment": normalized_scorer},
        callback=callback,
        logdir="results",
        save_interval=cfg.total_steps,
        experiment_name=f"{cfg.algorithm}_{cfg.dataset_name}_{cfg.seed}",
    )

    eval_rewards = [epoch[1]["environment"] for epoch in training_results]
    final_rew = normalized_scorer(model)
    max_rew = np.max(eval_rewards)
    std_rew = np.std(eval_rewards)

    print(f"Final reward: {final_rew:.4f}, best: {max_rew:.4f}, std: {std_rew:.4f}")
