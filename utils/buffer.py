from collections import deque
import gym

class Buffer: 

    def __init__(self, replay_buffer_size: int, min_replay_size: int, reward_buffer_size: int, env: gym.Env, fill: bool): 

        # Store attributes 
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_size = min_replay_size
        self.reward_buffer_size = reward_buffer_size
        self.env = env

        # Create buffers 
        self.replay_buffer = deque(maxlen = self.replay_buffer_size)
        self.reward_buffer = deque([0.0], maxlen = self.reward_buffer_size)

        # Fill Buffer 
        if fill: 
            obs = self.env.reset()
            for _ in range(self.min_replay_size): 
                action = self.env.action_space.sample()
                new_obs, rew, done, info = self.env.step(action)
                transition = (obs, action, rew, done, new_obs)
                self.replay_buffer.append(transition)
                obs = new_obs

                if done: 
                    obs = self.env.reset()
