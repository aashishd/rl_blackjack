import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import Env

RENDER_MODE = 'rgb_array'  # 'human'
RENDER = False


# from q_learning import run_q_learning


def run_black_jack():
    game_name = 'Blackjack-v1'
    lake_desc = None  # generate_random_map(3, 0.9) # None
    env: Env = gym.make(game_name, render_mode='rgb_array')  # generate_random_map(size=8)# g
    env.reset()
    env_screen = env.render()
    plt.imsave(f'games/{game_name}.png', env_screen)
    if RENDER:
        cv2.imshow('start', env_screen)
        cv2.waitKey(0)
    # run_policy_iteration(env, problem=f"{game_name}")
    # run_q_learning(env)


def get_blackjack_gym() -> Env:
    game_name = 'Blackjack-v1'
    env: Env = gym.make(game_name, render_mode='rgb_array')  # generate_random_map(size=8)# g
    env.reset()
    env_screen = env.render()
    plt.imsave(f'games/{game_name}.png', env_screen)
    if RENDER:
        cv2.imshow('start', env_screen)
        cv2.waitKey(0)
    return env
