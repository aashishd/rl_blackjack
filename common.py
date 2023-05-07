import json
import os.path

import numpy as np
import pandas as pd
from gymnasium.spaces import tuple

RUN_STATS_KEYS = ['State', 'Action', 'Reward', 'Error', 'Time', 'Max V', 'Mean V', 'Iteration']
METADATA_COLUMNS = ['Algo', 'Prob', 'Filepath', 'Metadata']

METADATA_FILENAME = '/Users/hellraizer/projects/rl-comparing_VI_PI_Q-Learning/stat_files_meatadata.csv'

# Frozen Lake
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP
FL_ACTION_MAPPING = ['<', 'v', '>', '^']


def save_stats(stats_df: pd.DataFrame, stats_dir: str, stats_metadata: dict, save_id=''):
    filename_builder = [save_id]
    for key, val in stats_metadata.items():
        if key != 'policy':
            filename_builder.append(f'{key}={val}')
    filename_builder.append('.csv')
    filepath = os.path.join(stats_dir, '_'.join(filename_builder))

    # save the stats
    stats_df.to_csv(filepath)

    # save the metadata
    meta_data_file = pd.DataFrame(columns=METADATA_COLUMNS)
    if os.path.isfile(METADATA_FILENAME):
        meta_data_file = pd.read_csv(METADATA_FILENAME, usecols=METADATA_COLUMNS)
    meta_data_file = meta_data_file.append({METADATA_COLUMNS[0]: stats_metadata['algo'], METADATA_COLUMNS[1]: stats_metadata['prob'],
                                            METADATA_COLUMNS[2]: filepath, METADATA_COLUMNS[3]: json.dumps(stats_metadata)}, ignore_index=True)
    meta_data_file.to_csv(METADATA_FILENAME)


def load_metadata_file():
    return pd.read_csv(METADATA_FILENAME, usecols=METADATA_COLUMNS)


def get_FL_policy_from_MDP_policy(mdp_policy):
    policy_arr = np.array(list(mdp_policy))
    reshpaed_size = round(np.sqrt(len(mdp_policy)))
    return policy_arr.reshape((reshpaed_size, reshpaed_size))


def print_FL_policy(policy: np.ndarray, env_desc):
    new_policy = np.zeros_like(policy, dtype=str)
    for i in range(4):
        new_policy[policy == i] = FL_ACTION_MAPPING[i]
    hole_indices = np.where(env_desc == b'H')
    new_policy[np.where(env_desc == b'H')] = 'O'
    new_policy[np.where(env_desc == b'G')] = 'G'
    print(new_policy)


def get_space_count(space):
    if type(space) == tuple.Tuple:
        return [obs.n for obs in space]
    return space.n
