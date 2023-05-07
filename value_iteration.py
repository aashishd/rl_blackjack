import pandas as pd
from hiive.mdptoolbox import mdp

from common import RUN_STATS_KEYS, save_stats, get_FL_policy_from_MDP_policy

VI_STATS_DIR = './vi_stats'
STATS_METADATA = dict(algo='VI')


def run_value_iteration(transition_matrix, reward_matrix, problem, gamma=0.9, epsilon=0.0001, save_id='', return_mdp=False):
    """
    runs value iteration and saves stats
    :param transition_matrix:
    :param reward_matrix:
    :param problem:
    :param epsilon:
    :param gamma:
    :return:
    """
    value_iteration = mdp.ValueIteration(transition_matrix, reward_matrix, gamma=gamma, epsilon=epsilon)
    value_iteration.run()
    FL_VI_stats_df = pd.DataFrame(data=value_iteration.run_stats, columns=RUN_STATS_KEYS)  # print(vi_fl)

    # parameters
    stats_metadata = STATS_METADATA.copy()
    stats_metadata['prob'] = problem
    stats_metadata['gamma'] = gamma
    stats_metadata['epsilon'] = epsilon
    stats_metadata['policy'] = str(list(value_iteration.policy))

    # save the stats of the run
    save_stats(FL_VI_stats_df, VI_STATS_DIR, stats_metadata, save_id)
    if return_mdp:
        return value_iteration, value_iteration.policy, FL_VI_stats_df
    return value_iteration.policy, FL_VI_stats_df
