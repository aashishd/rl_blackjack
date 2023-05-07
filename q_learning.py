# from q_learner import QLearningAgent
import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

from agent import QLearningAgent

# def learn_blackjack_using_qlearning():
#     # hyperparameters
#     learning_rate = 0.01
#     n_episodes = 100_000
#     start_epsilon = 1.0
#     epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
#     final_epsilon = 0.1
#
#     # Define the environment
#     game_name = 'Blackjack-v1'
#     env: Env = gym.make(game_name, render_mode='rgb_array')  # generate_random_map(size=8)# g
#     env.reset()
#     env_screen = env.render()
#
#     # Define the agent
#     agent = QLearningAgent(
#         env.action_space.n,
#         learning_rate=learning_rate,
#         initial_epsilon=start_epsilon,
#         epsilon_decay=epsilon_decay,
#         final_epsilon=final_epsilon,
#     )
#
#     env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
#     for episode in tqdm(range(n_episodes)):
#         obs, info = env.reset()
#         done = False
#
#         # play one episode
#         while not done:
#             action = agent.get_action(obs, env.action_space.sample())
#             next_obs, reward, terminated, truncated, info = env.step(action)
#
#             # update the agent
#             agent.update(obs, action, reward, terminated, next_obs)
#
#             # update if the environment is done and the current obs
#             done = terminated or truncated
#             obs = next_obs
#
#         agent.decay_epsilon()


# learn_blackjack_using_qlearning()


def train_QLearning_agent(agent: QLearningAgent, gym_env: Env, n_episodes=1000, max_iters=1500):
    """
    training loop for training a Q-Learning agent
    :param agent:
    :param gym_env:
    :param n_episodes:
    :param max_iters:
    :return: agent, env: tuple
    """
    env = gym.wrappers.RecordEpisodeStatistics(gym_env, deque_size=n_episodes)
    # training loop
    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        count = 0
        while not done:
            action = agent.get_action(obs, env.action_space.sample())
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            count += 1
            if count > max_iters:
                break
            obs = next_obs

        agent.decay_epsilon()

    return env
