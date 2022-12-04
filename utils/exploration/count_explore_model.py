from utils.exploration.base_exploration_model import BaseExplorationModel

class CountExploreModel(BaseExplorationModel):
    def __init__(self):
        super().__init__()
        self.words_tried = []

    def compute_bonus(self, board: dict, action):
        if action in self.words_tried:
            return -2
        else:
            self.words_tried.append(action)
            return -1