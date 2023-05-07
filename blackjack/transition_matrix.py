import pickle
from collections import defaultdict

import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm

BLACKJACK_STATE_DIMS = (32, 11, 2)
BLACKJACK_FLAT_DIMS = 32 * 11 * 2
BLACKJACK_ACTIONS_DIMS = 2

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def simulate_transition_matrix(n_episodes=1000_000):
    prob_dict = defaultdict(lambda: {a: {} for a in range(2)}) 
    env: Env = gym.make('Blackjack-v1', render_mode='rgb_array', sab=True)
    env.reset()
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        # play one episode
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            act_dict = prob_dict[obs][action]
            # update if the environment is done and the current obs
            done = terminated or truncated
            if done and action == 1 and next_obs[0] > 21 and reward < 0:
                # player goes bust
                next_obs = ('bust', next_obs[1], next_obs[2])
                # newstate, reward, terminated
            if done and action == 0:
                if reward == 0:
                    next_obs = 'tie'
                    # tie match
                elif reward > 0:
                    # player won
                    next_obs = 'win'
                elif reward < 0:
                    # player lost / dealer won
                    next_obs = 'loss'
            if (next_obs, reward, terminated) in act_dict.keys():
                act_dict[(next_obs, reward, terminated)] += 1
            else:
                act_dict[(next_obs, reward, terminated)] = 1

            obs = next_obs

    # convert occuring frequency to probabilities
    P = {}
    for start_state in prob_dict.keys():
        P[start_state] = {}
        for a in [0, 1]:
            trx_state_list = []
            end_states = prob_dict[start_state][a]
            tot_trxs = sum(end_states.values())
            for end_state, val in end_states.items():
                trx_prob = round(val / tot_trxs, ndigits=5)
                trx_state_list.append((trx_prob, end_state[0], end_state[1], end_state[2]))
            P[start_state][a] = trx_state_list

    with open('P_dict_simulated-test.pickle', 'wb') as f:
        pickle.dump(P, f)
    return P


if __name__ == '__main__':
    simulate_transition_matrix(100000)