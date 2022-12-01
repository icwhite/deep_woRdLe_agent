import stable_baselines3 as sb3
from utils.environments.wordle import Wordle
import os
import time

# Create Environment 
env = Wordle(n_boards = 1, 
             n_letters = 5, 
             n_guesses = 6)

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
# data_path = "Users/isado/cs285/cs285_final_project/data/"
logdir = "stable_baseline_dqn_no_fixed_answer" + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logging = os.path.join(data_path, logdir)

# Run DQN: Link to docs (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
agent = sb3.DQN(policy = 'MlpPolicy', 
                env = env, 
                learning_rate = 0.0003,
                buffer_size = 100_000,
                learning_starts = 50000, 
                batch_size = 64, 
                gamma = 0.99, 
                train_freq = (100, 'episode'), 
                gradient_steps = 1, 
                target_update_interval = 10000, 
                exploration_fraction = 0.5, # this number x total timesteps is the the total decay period
                exploration_initial_eps = 1.0, 
                exploration_final_eps = 0.001, 
                verbose = 1,
                tensorboard_log=logging)
agent.learn(total_timesteps = 1_000_000, log_interval = 1_000) # remember total times steps is number of guesses NOT number of games
