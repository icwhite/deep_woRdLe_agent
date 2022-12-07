# Idea: For first few games, just want to learn an implicit model of how likely you are to get a certain score
# when you guess a word.
from utils.exploration.base_exploration_model import BaseExplorationModel
from torch import nn
import utils.infrastructure.pytorch_utils as ptu
import torch.optim as optim
import torch.nn.functional as Fun
import numpy as np

class DiffPrevWordsExplore(BaseExplorationModel):
    """
    Give bonus based on how many different letters have been tried so far.
    bonus is letters_tried divided by total number of letters entered.
    """

    def compute_bonus(self, boards: list, action: str, guess_count: list):
        s = 0
        for i in range(len(boards)):
            s += self._compute_single_board_bonus(boards[i], action, guess_count[i])
        return s

    def _compute_single_board_bonus(self, board: dict, action: str, guess_count: int):
        state = np.array(board["letters"])
        state = state.flatten()
        letters_tried = 0
        so_far = np.zeros(26)

        for number in state:
            if not so_far[number] == 1:
                letters_tried += 1
                so_far[number] = 1
        return letters_tried / (guess_count * 5)