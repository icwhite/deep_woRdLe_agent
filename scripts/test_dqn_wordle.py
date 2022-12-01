# Dependencies 
import torch
from torch import nn
import numpy as np
import random

from utils.environments.wordle import Wordle
from utils.policies.mlp import MLPPolicy
from utils.infrastructure.buffer import Buffer
import utils.infrastructure.pytorch_utils as ptu
from utils.agents.dqn_agent import DQNAgent
import os
import time

import gym


# Network Parameters 
network_params = {'n_layers': 2,
                  'size': 32,
                  'activation': nn.ReLU(),
                  'output_activation': nn.Identity(),
                  'lr': 5e-2,
                  'batch_size': 32
                  }

# Buffer Parameters 
buffer_params = {'replay_buffer_size': 100_000,
                 'reward_buffer_size': 100,
                 'min_replay_size': 1_000
                 }

# Exploration Parameters
exploration_params = {'epsilon_start': 1.0,
                      'epsilon_end': 0.02,
                      'epsilon_decay': 300_000
                      }

# RL Parameters 
rl_params = {'n_iter': 1_000_000,
             'log_period': 1000,
             'num_eval_episodes': 100,
             'gamma': 0.99,
             'target_update_freq': 1_000
             }

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

logdir = f"dqn_wordle" + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

logging_params = {
    "logdir": os.path.join(data_path, logdir)
}

# Create environment 
env = Wordle(n_boards = 1,
             n_letters = 5,
             n_guesses = 6)



# Combine Parameters
params = {k: v for d in [logging_params, network_params, buffer_params, exploration_params, rl_params] for k, v in d.items()}

# Construct agent 
agent = DQNAgent(env, **params)
agent.train_agent()


