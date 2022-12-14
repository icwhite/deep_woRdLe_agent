import stable_baselines3 as sb3
from utils.environments.wordle import Wordle
from utils.exploration.count_explore_model import CountExploreModel
from utils.exploration.diff_prev_words import DiffPrevWordsExplore

import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="dqn_wordle")
parser.add_argument("--subset_valid_words", type=int, default=0)
parser.add_argument("--subset_answers", type=int, default=0)
parser.add_argument("--timesteps", type=int, default= 1_000_000)
parser.add_argument("--explore", type=str, default="diff")
parser.add_argument("--reward", type=str, default="elimination")


args = parser.parse_args()
params = vars(args)

wordle_words = open("scripts/wordle_words.txt", "r").read().split(",")
wordle_words = [word.replace('\n', '') for word in wordle_words]

if params["explore"] == "diff":
    exploration_model = DiffPrevWordsExplore()
elif params["explore"] == "count":
    exploration_model = CountExploreModel()



data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
# data_path = "Users/isado/cs285/cs285_final_project/data/"
logdir = "ppo" + "_" + params["exp_name"] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logging = os.path.join(data_path, logdir)

# Create Environment
env = Wordle(n_boards=1,
             n_letters=5,
             n_guesses=6,
             subset_valid_words=params["subset_valid_words"],
             subset_answers=params["subset_answers"],
             keep_answers_on_reset=False,
             exploration_model= exploration_model,
             valid_words=wordle_words,
             logdir=os.path.join(logging, "win_logs"))

# Run DQN: Link to docs (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
agent = sb3.PPO(policy = 'MlpPolicy',
                env = env,
                learning_rate = 0.0003,
                n_steps = 2048,
                batch_size = 64,
                n_epochs = 10,
                gamma = 0.99,
                gae_lambda = 0.95,
                clip_range = 0.2,
                verbose = 1,
                tensorboard_log=logging)
agent.learn(total_timesteps=params["timesteps"], log_interval=4) # remember total times steps is number of guesses NOT number of games
agent.save(params["exp_name"])
