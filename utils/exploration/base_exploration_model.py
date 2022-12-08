from collections import deque
import numpy as np

class BaseExplorationModel(object):

    def __init__(self):
        self.prev_bonuses = deque(maxlen=500)
        # self.gamma = 1
        self.t = 0


    def compute_bonus(self, state: dict, action: str, guess_count: int):
        return 0

    def compute_normalized_bonus(self, state:dict, action:str, guess_count:int):
        # gamma = self.rnd_gamma / (1 + self.t)
        bonus = self.compute_bonus(state, action, guess_count)

        self.prev_bonuses.append(bonus)

        # self.running_rnd_rew_std = gamma * std_dev + (1 - gamma) * self.running_rnd_rew_std
        std_dev = np.std(self.prev_bonuses)
        mean = np.mean(self.prev_bonuses)
        self.t += 1

        return (bonus - mean)/std_dev