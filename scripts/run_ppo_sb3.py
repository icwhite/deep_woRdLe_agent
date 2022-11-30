import stable_baselines3 as sb3
from utils.environments.wordle import Wordle

# Create Environment 
env = Wordle(n_boards = 1, 
             n_letters = 5, 
             n_guesses = 6, 
             answers = ['train'], 
             keep_answers_on_reset = True)

# Run DQN: Link to docs (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
agent = sb3.PPO(policy = 'MlpPolicy', 
                env = env, 
                learning_rate = 0.0003, 
                n_steps = 2048, 
                batch_size = 64, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lambda = 0.95, 
                clip_range = 0.2, 
                verbose = 1)
agent.learn(total_timesteps = 1_000_000, log_interval = 4) # remember total times steps is number of guesses NOT number of games
