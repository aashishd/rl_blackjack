import json
from time import time

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.blackjack import BlackjackEnv

from agent import QLearningAgent
from blackjack.blackjack_util import get_policy_from_qtable, get_mdp_policy_from_gym_policy, get_values_from_qtable, map_to_state_indexes
from blackjack.blackjack_util import play_using_policy, get_gym_policy_from_mdp_policy
from blackjack.visualize_policy import get_policy_grid, create_plots, create_grids
from q_learning import train_QLearning_agent


def train_agent_on_blackjack(FL_ENV, learning_rate=0.01, n_episodes=100000, start_epsilon=1.0, final_epsilon=0.1, gamma=0.95, decay='linear'):
    if decay == 'linear':
        epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    elif decay == 'exp':
        epsilon_decay = 0.99

    # Define the agent
    agent = QLearningAgent(
        FL_ENV.action_space.n,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=gamma,
        decay=decay,
        init_q_values='zero'
    )

    env_wrapper = train_QLearning_agent(agent, FL_ENV, n_episodes=n_episodes)

    return agent, env_wrapper


def get_policy_from_q_learning(q_table, states):
    return [np.argmax(q_table[s]) for s in states]


QL_metatdata_columns = ['Episodes', 'Params', 'Policy', 'Time', 'Reward Queue', 'Episode_Length Queue', 'Training Error', 'Time Delta']

if __name__ == '__main__':
    LR = [0.01]  # [0.01, 0.001, 0.1, 0.5]
    DECAY = ['linear']  # ['linear', 'exp']
    GAMMAS = [0.8]  # [0.8, 0.95, 0.99]
    n_episodes = 150_000
    blackjack_env = BlackjackEnv()
    runs_dict = {}
    for learning_rate in LR:
        for decay in DECAY:
            for gamma in GAMMAS:
                start_epsilon = 1.0
                final_epsilon = 0.1
                start_time = time()
                agent, env_wrapper = train_agent_on_blackjack(blackjack_env, learning_rate, n_episodes, start_epsilon, final_epsilon, gamma=gamma, decay=decay)
                exec_time = time() - start_time
                policy = get_policy_from_qtable(agent.q_values)
                values = get_values_from_qtable(agent.q_values)
                values_mdp = map_to_state_indexes(values)
                mdp_policy = get_mdp_policy_from_gym_policy(policy)
                stats_dict = {
                    QL_metatdata_columns[0]: n_episodes,
                    QL_metatdata_columns[1]: dict(learning_rate=learning_rate, n_episodes=n_episodes, start_epsilon=start_epsilon, final_epsilon=final_epsilon),
                    QL_metatdata_columns[2]: mdp_policy,
                    QL_metatdata_columns[3]: exec_time,
                    QL_metatdata_columns[4]: [int(i[0]) for i in env_wrapper.return_queue],
                    QL_metatdata_columns[5]: [int(i[0]) for i in env_wrapper.length_queue],
                    QL_metatdata_columns[6]: list(map(int, agent.training_error)),
                    QL_metatdata_columns[7]: list(map(float, agent.episode_time_delta))
                }
                filename = f'./ql_stats/Blackjack-lr_{learning_rate}-gamma_{gamma}-decay_{decay}'
                with open(f'{filename}.json', 'w') as f:
                    f.write(json.dumps(stats_dict))

                print(f'\n******************* Game plays for lr_{learning_rate}-gamma_{gamma}-decay_{decay} *******************')

                policy_grid, values_grid = create_grids(agent, usable_ace=False)
                fig1 = create_plots(values_grid, policy_grid, rf'Q-Learning Policy for Blackjack without usable Ace $\gamma$={gamma} & $\alpha$={learning_rate}')
                plt.show()

                policy_grid, values_grid = create_grids(agent, usable_ace=True)
                fig2 = create_plots(values_grid, policy_grid, rf'Q-Learning Policy for Blackjack with usable Ace $\gamma$={gamma} & $\alpha$={learning_rate}')
                plt.show()
                # runs_dict[gamma] = (policy_grid, vi_stats)
                play_using_policy(BlackjackEnv(), policy, games=1000)
