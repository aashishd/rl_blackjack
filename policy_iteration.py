import pandas as pd
from hiive.mdptoolbox import mdp

from common import RUN_STATS_KEYS, save_stats, get_FL_policy_from_MDP_policy

VI_STATS_DIR = './pi_stats'
STATS_METADATA = dict(algo='PI')


def run_policy_iteration(transition_matrix, reward_matrix, problem, gamma, save_id='', return_mdp=False):
    """
    runs Policy iteration and saves stats
    :param transition_matrix:
    :param reward_matrix:
    :param problem:
    :param gamma:
    :return:
    """
    policy_iteration = mdp.PolicyIteration(transition_matrix, reward_matrix, gamma=gamma)
    policy_iteration.run()
    FL_VI_stats_df = pd.DataFrame(data=policy_iteration.run_stats, columns=RUN_STATS_KEYS)  # print(vi_fl)

    # parameters
    stats_metadata = STATS_METADATA.copy()
    stats_metadata['prob'] = problem
    stats_metadata['gamma'] = gamma
    stats_metadata['policy'] = str(list(policy_iteration.policy))

    # save the stats of the run
    save_stats(FL_VI_stats_df, VI_STATS_DIR, stats_metadata, save_id)
    if return_mdp:
        return policy_iteration, policy_iteration.policy, FL_VI_stats_df
    return policy_iteration.policy, FL_VI_stats_df
