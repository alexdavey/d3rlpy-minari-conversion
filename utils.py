import minari
import numpy as np
from d3rlpy.dataset import MDPDataset
from gymnasium.wrappers import FlattenObservation, TimeLimit


def minariTod3rl(minari_dataset):
    """Convert between a Minari dataset and the MDPDataset used by d3rlpy. """
    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    for episode in minari_dataset:
        # Minari datasets include the observation of the final state,
        # whereas MDPDataset does not, so we remove the last observation
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        terminations.append(episode.terminations)
        truncations.append(episode.truncations)

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminations = np.concatenate(terminations, axis=0)
    truncations = np.concatenate(truncations, axis=0)

    # MDPDataset only supports terminations, whereas Minari supports both
    # termations and truncations, which we OR together
    term_or_trunc = terminations | truncations

    print(f"SUM rewards: {np.sum(rewards)}")
    return MDPDataset(
        observations, actions, rewards, terminations, episode_terminals=term_or_trunc
    )


# Needed to wrap this, since it does not support dicts
def evaluate_minari_policy(env, n_trials=10, epsilon=0.0):
    def scorer(algo, *args) -> float:
        episode_rewards = []
        for _ in range(n_trials):
            observation, _ = env.reset()
            episode_reward = 0.0

            step = 0

            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict([observation])[0]

                obs, rew, term, trunc, info = env.step(action)
                episode_reward += rew

                if term or trunc:
                    break

                step += 1

            episode_rewards.append(episode_reward)
        print(episode_rewards)
        print(f"=> {np.mean(episode_rewards)}")
        return float(np.mean(episode_rewards))

    return scorer


def get_MDPDatasetEnv(dataset_name, is_minari_dataset, max_episode_steps):
    """Get a d3rlpy dataset, either directly, or by converting a Minari one."""
    # If it's not a Minari dataset, assume it's the original D4RL dataset
    if not is_minari_dataset:
        return d3rlpy.datasets.get_d4rl(dataset_name)

    # Check if the dataset exists locally, or else download it
    if dataset_name not in minari.list_local_datasets():
        if dataset_name in minari.list_remote_datasets():
            minari.download_dataset(dataset_name)
        else:
            raise RuntimeError(
                f'No local or remote dataset named "{dataset_name}" found.'
            )

    # Load the dataset and environment from Minari
    # The Minari envs returned do not always have a useable
    # TimeLimit wrapped around it, so we override it
    minari_dataset = minari.load_dataset(dataset_name)
    env = minari_dataset.recover_environment()
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Convert to the MDPDataset format supported by d3rlpy
    dataset = minariTod3rl(minari_dataset)

    return dataset, env
