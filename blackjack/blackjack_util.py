import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import Env
from tqdm import tqdm

from gym_to_mdp import get_P_and_R_matrices

BLACKJACK_STATE_DIMS = (32, 11, 2)
BLACKJACK_FLAT_DIMS = 32 * 11 * 2
BLACKJACK_ACTIONS_DIMS = 2

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# convert to pseudocode
def simulate_transition_matrix(n_episodes=1000000):
    prob_dict = defaultdict(lambda: {0: {}, 1: {}})
    env: Env = gym.make('Blackjack-v1', render_mode='rgb_array', sab=True)
    # play one game of blackjack
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        # play one episode
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            act_dict = prob_dict[obs][action]
            # update if the environment is done and the current obs
            done = terminated or truncated
            if done and action == 1 and next_obs[0] > 21 and reward < 0:
                # player goes bust
                next_obs = ('bust', next_obs[1], next_obs[2])
                # newstate, reward, terminated
            if done and action == 0:
                if reward == 0: # tie match
                    next_obs = 'tie'
                elif reward > 0: # player won
                    next_obs = 'win'
                elif reward < 0: # player lost / dealer won
                    next_obs = 'loss'
            if (next_obs, reward, terminated) in act_dict.keys():
                act_dict[(next_obs, reward, terminated)] += 1
            else:
                act_dict[(next_obs, reward, terminated)] = 1
            obs = next_obs

    # convert to probabilities
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

    with open('P_dict_simulated.pickle', 'wb') as f:
        pickle.dump(P, f)
    return P


def prob_player_winning(cur_state, dealer_exp_scores_df):
    p_win, p_tie, p_loss = 0, 0, 0
    p_score, d_face, u_ace = cur_state
    if p_score > 21:
        return 0, 0, 1
    if p_score < 17:
        p_win = dealer_exp_scores_df.at[d_face, 'bust']
        p_loss = 1. - p_win
    else:
        p_tie = dealer_exp_scores_df.at[d_face, f'{p_score}']
        p_win = sum([dealer_exp_scores_df.at[d_face, f'{ds}'] for ds in range(17, p_score)])
        p_loss = 1. - (p_win + p_tie)
    return p_win, p_tie, p_loss


def update_P_dict_to_indices_dict():
    with open('P_dict_simulated.pickle', 'rb') as f:
        P_dict = pickle.load(f)

        # sorted_keys = sorted(list(P_dict.keys()), key=lambda x: (x[0], x[1], int(x[2])))
        # Path('sorted_keys.json').write_text(json.dumps(sorted_keys))
    sorted_keys: list = json.loads(Path('sorted_keys.json').read_text())

    P = {}
    for start_state in P_dict.keys():
        s = sorted_keys.index(list(start_state))
        P[s] = {}
        for a in [0, 1]:
            trx_state_list = []
            end_states = P_dict[start_state][a]
            # tot_trxs = sum(end_states.values())
            for trx_prob, end_state, reward, terminated in end_states:
                s_next = None
                if end_state[0] == 'bust':
                    s_next = sorted_keys.index('bust')
                elif end_state in ['win', 'loss', 'tie']:
                    s_next = sorted_keys.index(end_state)
                else:
                    s_next = sorted_keys.index(list(end_state))
                trx_state_list.append((trx_prob, s_next, reward, terminated))
            P[s][a] = trx_state_list

    # add absorbing states
    for s in [280, 281, 282, 283]:
        P[s] = {}
        for a in [0, 1]:
            P[s][a] = [(1.0, s, 0.0, True)]

    with open('P_blackjack.pickle', 'wb') as f:
        pickle.dump(P, f)


def load_P_dict():
    with open('P_blackjack.pickle', 'rb') as f:
        return pickle.load(f)


def convert_policy_to_grid(policy):
    sorted_keys: list = json.loads(Path('sorted_keys.json').read_text())


def get_sorted_keys() -> List:
    return json.loads(Path('sorted_keys.json').read_text())


def pickle_mdp_model(filepath, mdp):
    with open(filepath, 'wb') as f:
        return pickle.dump(mdp, f)


def get_gym_policy_from_mdp_policy(policy_mdp):
    policy = defaultdict(int)
    state_value = defaultdict(int)
    state_idx_mapping = get_sorted_keys()
    for idx, action in enumerate(policy_mdp[:280]):
        policy[tuple(state_idx_mapping[idx])] = action
    return policy


def get_mdp_policy_from_gym_policy(policy_gym):
    policy = [None for _ in range(280)]
    state_value = defaultdict(int)
    state_idx_mapping = get_sorted_keys()
    for state, action in policy_gym.items():
        state = list(state)
        if state in state_idx_mapping:
            policy[state_idx_mapping.index(state)] = action
    return policy


def get_values_from_qtable(q_table):
    policy = {}
    for state, action_vals in q_table.items():
        policy[state] = np.max(action_vals)
    return policy


def map_to_state_indexes(state_vs_vals):
    policy = [None for _ in range(280)]
    state_value = defaultdict(int)
    state_idx_mapping = get_sorted_keys()
    for state, value in state_vs_vals.items():
        state = list(state)
        if state in state_idx_mapping:
            policy[state_idx_mapping.index(state)] = value
    return policy


def get_policy_from_qtable(q_table):
    policy = {}
    for state, action_vals in q_table.items():
        policy[state] = int(np.argmax(action_vals))
    return policy


def state_to_idx(state):
    return np.ravel_multi_index(([state[0]], [state[1]], [int(state[2])]), dims=BLACKJACK_STATE_DIMS)[0]


def idx_to_state(idx):
    state_tup = np.unravel_index([idx], shape=BLACKJACK_STATE_DIMS)
    return state_tup[0][0], state_tup[1][0], state_tup[2][0]


def play_using_policy(env, policy, games=200, max_tries=100, log=True):
    """Simulate a gameplay using the policy"""
    rewards_queue = []
    for _ in tqdm(range(games)):
        # print('playing')
        obs, info = env.reset()
        done = False

        # play one episode
        count = 0
        while not done:
            action = policy[obs]
            next_obs, reward, terminated, truncated, info = env.step(action)
            # update if the environment is done and the current obs
            # print(f"next_obs = {next_obs} ; action = {action}")
            done = terminated or truncated
            obs = next_obs
            count += 1
            if done:
                rewards_queue.append(reward)
            if count > max_tries:
                rewards_queue.append('T')
                break
    success_pct = (rewards_queue.count(1) / games) * 100
    not_finish_pct = (rewards_queue.count('T') / games) * 100
    if log:
        print(f'Success % => {success_pct} %')
        print(f"Didn't finish % => {not_finish_pct} %")
    return rewards_queue, success_pct, not_finish_pct

# if __name__ == '__main__':
#     update_P_dict_to_indices_dict()
# P = load_P_dict()
# P_dict = generate_prob_matrix()
# P, R = get_P_and_R_matrices(P_dict, [0, 1], list(P_dict.keys()))
# P_dict = simulate_transition_matrix()
