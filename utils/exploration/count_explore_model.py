from utils.exploration.base_exploration_model import BaseExplorationModel
import numpy as np

class CountExploreModel(BaseExplorationModel):
    def __init__(self):
        super().__init__()
        self.words_tried = {}

    def compute_bonus(self, state, action, guess_count):
        if action in self.words_tried.keys():
            self.words_tried[action] += 1
        else:
            self.words_tried[action] = 1
        return 1 / np.sqrt(self.words_tried[action])