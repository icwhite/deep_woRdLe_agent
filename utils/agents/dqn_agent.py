import numpy as np
import torch
from torch import nn
import random

from policies.mlp import MLPPolicy
from infrastructure.buffer import Buffer
import infrastructure.pytorch_utils as ptu
from infrastructure.logger import Logger


class DQNAgent:

    def __init__(self, env, **params):

        # Grab attributes from environment
        self.env = env
        obs_dim = np.prod(env.state.shape, dtype=int)
        ac_dim = env.action_space.n
        self.params = params

        # Create Online and Target networks
        self.online_net = MLPPolicy(obs_dim, ac_dim, params['n_layers'], params['size'], params['activation'],
                                    params['output_activation'])
        self.target_net = MLPPolicy(obs_dim, ac_dim, params['n_layers'], params['size'], params['activation'],
                                    params['output_activation'])

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=params['lr'])

        self.logger = Logger(self.params["logdir"])

        # Initialize replay buffers and fill
        self.buffer = Buffer(params['replay_buffer_size'], params['min_replay_size'], params['reward_buffer_size'], env,
                             fill=True)
        self.replay_buffer = self.buffer.replay_buffer
        self.reward_buffer = self.buffer.reward_buffer

    def evaluate_agent(self):
        num_episodes = 0
        eval_rewards = []
        obs = self.env.reset()
        episode_reward = 0
        while num_episodes < self.params['num_eval_episodes']:
            action = self.online_net.get_action(obs)
            # Take the action and store transition
            next_obs, rew, done, info = self.env.step(action)
            obs = next_obs
            episode_reward += rew

            # Reset the environment if necessary
            if done:
                obs = self.env.reset()
                eval_rewards.append(episode_reward)
                episode_reward = 0
                num_episodes += 1

        average_eval_reward = np.mean(eval_rewards)
        std_eval_reward = np.std(eval_rewards)

        return average_eval_reward, std_eval_reward

    def train_agent(self):

        obs = self.env.reset()
        num_episodes = 0
        episode_reward = 0
        for iter in range(self.params['n_iter']):
            new_episode = False
            # Compute epsilon 
            epsilon = np.interp(iter, [0, self.params['epsilon_decay']],
                                [self.params['epsilon_start'], self.params['epsilon_end']])
            random_val = random.random()

            # Take random action with probability epsilon
            if random_val <= epsilon:
                action = self.env.action_space.sample()

            else:
                # Otherwise use the online network to take the action
                action = self.online_net.get_action(obs)

            # Take the action and store transition
            next_obs, rew, done, info = self.env.step(action)
            transition = (obs, action, rew, done, next_obs)
            self.replay_buffer.append(transition)
            obs = next_obs

            episode_reward += rew

            # Reset the environment if necessary
            if done:
                obs = self.env.reset()
                self.reward_buffer.append(episode_reward)
                episode_reward = 0
                num_episodes += 1
                new_episode = True

            # Sample transitions from the replay buffer
            transitions = random.sample(self.replay_buffer, self.params['batch_size'])

            # Convert everything to tensors
            obses_t = ptu.from_numpy(np.array([t[0] for t in transitions], dtype=np.int64))
            actions_t = ptu.from_numpy(np.array([t[1] for t in transitions], dtype=np.int64)).unsqueeze(-1)
            rewards_t = ptu.from_numpy(np.array([t[2] for t in transitions]))
            dones_t = ptu.from_numpy(np.array([t[3] for t in transitions]))
            next_obses_t = ptu.from_numpy(np.array([t[4] for t in transitions], dtype=np.int64))

            # Fix datatypes
            obses_t = obses_t.to(torch.int64)
            next_obses_t = next_obses_t.to(torch.int64)
            actions_t = actions_t.to(torch.int64)

            # Compute targets
            next_q = self.target_net(next_obses_t)
            target_q = (next_q.max(dim=1, keepdim=True)[0]).squeeze(-1)
            targets = rewards_t + self.params['gamma'] * (1 - dones_t) * target_q  # 32, 32

            # Compute loss 
            current_q = self.online_net(obses_t)
            action_q_values = torch.gather(input=current_q, dim=1, index=actions_t).squeeze(-1)  # 32,1
            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Step Optimizer 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target network 
            if iter % self.params['target_update_freq'] == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # Logging
            # log_period now refers to number of episodes
            if new_episode and num_episodes % self.params['log_period'] == 0:
                average_eval_reward, std_eval_reward = self.evaluate_agent()
                logs = {
                    "Average Train Reward": np.mean(self.reward_buffer),
                    "Std Train Reward": np.std(self.reward_buffer),
                    "Average Eval Reward": average_eval_reward,
                    "Std Eval Reward": std_eval_reward,
                    "Training Loss": loss
                }
                self.do_logging(logs, num_episodes, iter)
    
    def do_logging(self, logs, num_episodes, step):
        print(f"Episode: {num_episodes}")
        print(f"Iteration: {step}")
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, step)
        print("\n")
