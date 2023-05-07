from blackjack.blackjack_util import load_P_dict, pickle_mdp_model, play_using_policy, get_gym_policy_from_mdp_policy
from gym_to_mdp import get_P_and_R_matrices
from value_iteration import run_value_iteration
from blackjack.visualize_policy import get_policy_grid, create_plots
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
import matplotlib.pyplot as plt

GAME_NAME = 'BlackJack'

if __name__ == '__main__':
    # FL_GYM, FL_MDP = get_frozen_lake_env()
    P = load_P_dict()
    ALL_STATES = list(range(284))
    ALL_ACTIONS = [0, 1]
    T, R = get_P_and_R_matrices(P, ALL_ACTIONS, ALL_STATES, normalize=True)
    runs_dict = {}

    for gamma in [0.7, 0.8, 0.95, 0.99]:
        filepath = f'vi_stats/Blackjack-VI-gamma_{gamma}.pickle'
        mdp, policy_mdp, vi_stats = run_value_iteration(T, R, GAME_NAME, gamma=gamma, epsilon=1e-15, return_mdp=True)
        print(f'\n*********************Stats for gamma = {gamma} *********************\n')
        pickle_mdp_model(filepath, mdp)

        policy_grid, values_grid = get_policy_grid(policy_mdp, mdp.V)
        fig1 = create_plots(values_grid, policy_grid, rf'VI Policy for Blackjack without usable Ace $\gamma$={gamma}')
        plt.show()

        policy_grid, values_grid = get_policy_grid(policy_mdp, mdp.V, usable_ace=True)
        fig2 = create_plots(values_grid, policy_grid, rf'VI Policy for Blackjack with usable Ace $\gamma$={gamma}')
        plt.show()
        # runs_dict[gamma] = (policy_grid, vi_stats)
        play_using_policy(BlackjackEnv(), get_gym_policy_from_mdp_policy(policy_mdp), games=1000)
