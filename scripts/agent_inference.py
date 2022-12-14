import stable_baselines3 as sb3
from utils.environments.wordle import Wordle
import os
import time
import argparse

# Get wordle words
wordle_words = open("scripts/wordle_words.txt", "r").read().split(",")
wordle_words = [word.replace('\n', '') for word in wordle_words]

# Create Environment
env = Wordle(n_boards=1,
             n_letters=5,
             n_guesses=6,
             answers=['least'],
             valid_words=wordle_words,
            keep_answers_on_reset=True)

model = sb3.PPO.load("saved_agents/ppo_lr_0.0003_2mill")
# Enjoy trained agent
obs = env.reset()
print(env.answers)
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(env.valid_words[action])
    print(env.possible_words)
