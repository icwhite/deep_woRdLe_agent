# Dependencies 
import torch
from torch import nn
import numpy as np
import random

from environments.wordle import Wordle
from policies.mlp import MLPPolicy
from utils.buffer import Buffer
import utils.pytorch_utils as ptu
from agents.dqn_agent import DQNAgent
import os
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--letters", type=int, default=5)
parser.add_argument("--exp_name", type=str, default="dqn_wordle")
parser.add_argument("--fixed_answer", action="store_true")

args = parser.parse_args()
params = vars(args)


# Wordle Parameters 
wordle_params = {'n_boards': 1, 
                 'n_letters': params["letters"],
                 'n_guesses': 6,
                 'fixed_answer': params["fixed_answer"]
                 }

# Network Parameters 
network_params = {'n_layers': 2,
                  'size': 128,
                  'activation': nn.ReLU(),
                  'output_activation': nn.Identity(),
                  'lr': 5e-4,
                  'batch_size': 32
                  }

# Buffer Parameters 
buffer_params = {'replay_buffer_size': 50_000,
                 'reward_buffer_size': 100,
                 'min_replay_size': 1_000
                 }

# Exploration Parameters
exploration_params = {'epsilon_start': 1.0,
                      'epsilon_end': 0.02,
                      'epsilon_decay': 10_000
                      }

# RL Parameters 
rl_params = {'n_iter': 1_000_000,
             'log_period': 1_000,
             'num_eval_episodes': 100,
             'gamma': 0.99,
             'target_update_freq': 1_000
             }

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
logdir = params["exp_name"] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

logging_params = {
    "logdir": os.path.join(data_path, logdir)
}

# Create environment 
env = Wordle(**wordle_params)
print(f'Playing Wordle with {env.n_boards} Boards, {env.n_letters} letters and {env.n_guesses} guesses.')
print(f'There are {len(env.valid_words)} playable words in this configuration.')


# Combine Parameters
params = {k: v for d in [logging_params, network_params, buffer_params, exploration_params, rl_params] for k, v in d.items()}

# Construct agent 
agent = DQNAgent(env, **params)
agent.train_agent()


