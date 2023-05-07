import re

import numpy as np
from gymnasium import Env


class GymToMDPConverter:
    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments.

    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control

    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.  
    """

    def __init__(self, gym_env):
        """Create a new instance of the OpenAI_MDPToolbox class

        :param openAI_env_name: Valid name of an Open AI Gym env 
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean 
        """
        self.env: Env = gym_env
        self.transitions = self.env.P
        # Each value in P contains => probability, newstate, reward, terminated
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        # self.R = np.zeros(self.states)
        self._convert_PR()

    def _convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob, state_, reward, is_terminal = self.transitions[state][action][i]
                    self.P[action, state, state_] += tran_prob
                    self.R[state][action] += tran_prob * reward
                    # self.R[state_] = self.transitions[state][action][i][2]


def get_P_and_R_matrices(gym_transitions, all_actions, all_states, normalize=False):
    action_count = len(all_actions)
    states_count = len(all_states)
    P = np.zeros((action_count, states_count, states_count))
    R = np.zeros((states_count, action_count))
    for state in all_states:
        for action in all_actions:
            for i in range(len(gym_transitions[state][action])):
                tran_prob, state_, reward, is_terminal = gym_transitions[state][action][i]
                P[action, state, state_] += tran_prob
                R[state][action] += tran_prob * reward

    if normalize:
        for a in all_actions:
            row_sum = np.sum(P[a], axis=1)
            P[a] = P[a] / row_sum[:, np.newaxis]
    return P, R


def test_frozen_lake():
    fl_mdp = GymToMDPConverter('FrozenLake-v1', desc=None, is_slippery=True, render_mode='human', render=True)
    print(fl_mdp.P.shape)
    print(fl_mdp.R.shape)
