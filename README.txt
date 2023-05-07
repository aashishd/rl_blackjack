#Blackjack
##Step 1 : To run the all the experiments can be run using the run files located in the modules specific to the MDPs
1. Blackjack
- blackjack.run_BJ_policy_iteration.py
- blackjack.run_BJ_value_iteration.py
- blackjack.run_BJ_q_learning.py

All the above files run the code specific to the algorithm. The hyper-parameters can be changed within the module.


##Step 2: Once the algorithms executed, the necessary stats and policy images will be stored in the vi_stats, pi_stats, ql_stats directories for the respective algorithms.
These can now be used to generate various plots using the two include jupyer notebooks as follows :
- frozen_lake_analysis.ipynb
- blackjack_analysis.ipynb

##Other files
agent.py : Contains the QLearningAgent which can be used for both frozen lake and Blackjack
common.py : Some common functions for frozen lake stats and policy generation
gym_to_mdp.py : Contains code to convert the Gym transition matrices to MDP toolbox and vice versa
policy_iteration.py : Wrapper around MDPToolbox policy iteration which also saves the stats
value_iteration.py : Wrapper around MDPToolbox value iteration which also saves the stats
q_learning.py : Q-Learning training loop where agent interacts with the environment
blackjack.blackjack_util.py : Utility functions for building the Model for Blackjack and simulating the gameplay
blackjack.visualize_policy.py : Visualize the policy for the Blackjack game
frozen_lake.frozen_lake_game.py : Simulation of Frozen lake game and getting the environment with parametes
frozen_lake.visualize_policy.py : Visualize the policy for the FrozenLake game


##Folowing python libraries are used only :
 - matplotlib
 - gymnasium
 - mdptoolbox
 - seaborn
 - numpy
 - scipy
 - tqdm
 - pandas

