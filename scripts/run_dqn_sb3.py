import stable_baselines3 as sb3
from utils.environments.wordle import Wordle

# Create Environment 
env = Wordle(n_boards = 1, 
             n_letters = 5, 
             n_guesses = 6, 
             answers = ['train'], 
             keep_answers_on_reset = True)

# Run DQN: Link to docs (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
agent = sb3.DQN(policy = 'MlpPolicy', 
                env = env, 
                learning_rate = 1e-4, 
                buffer_size = 1_000_000, 
                learning_starts = 50000, 
                batch_size = 64, 
                gamma = 0.99, 
                train_freq = (100, 'episode'), 
                gradient_steps = 1, 
                target_update_interval = 10000, 
                exploration_fraction = 0.1, 
                exploration_initial_eps = 1.0, 
                exploration_final_eps = 0.05, 
                verbose = 1,)
agent.learn(total_timesteps = 1_000_000, log_interval = 4) # remember total times steps is number of guesses NOT number of games
