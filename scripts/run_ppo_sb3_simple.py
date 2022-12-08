import stable_baselines3 as sb3
from utils.environments.wordle import WordleSimple
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="simple")
parser.add_argument("--learning_rate", '-lr', type=float, default = 0.0003)
parser.add_argument("--timesteps", type=int, default= 1_000_000)



args = parser.parse_args()
params = vars(args)

# Get wordle words
wordle_subset = open("scripts/wordle_subset.txt", "r").read().split(",")
wordle_subset = [word.replace('\n', '') for word in wordle_subset]

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
logdir = "ppo_sb3" + "_" + params["exp_name"] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logging = os.path.join(data_path, logdir)

# Create Environment
env = WordleSimple(valid_words = wordle_subset, keep_answer_on_reset = False, logdir = os.path.join(logging, "win_logs"))


agent = sb3.PPO(policy = 'MlpPolicy',
                env = env, 
                learning_rate = params['learning_rate'], 
                n_steps = 2048, 
                batch_size = 64, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lambda = 0.95, 
                clip_range = 0.2, 
                verbose = 1,
                tensorboard_log=logging)
agent.learn(total_timesteps = params["timesteps"], log_interval = 1) # remember total times steps is number of guesses NOT number of games
agent.save(params["exp_name"] + "_" + str(params["timesteps"]))
