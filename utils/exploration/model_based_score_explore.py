# Idea: For first few games, just want to learn an implicit model of how likely you are to get a certain score
# when you guess a word.
from utils.exploration.base_exploration_model import BaseExplorationModel
from torch import nn
import utils.infrastructure.pytorch_utils as ptu
import torch.optim as optim
import torch.nn.functional as Fun

class ModelScoreExplore(BaseExplorationModel):

    def __init__(self, n_layer=2, size=128, board_size=30, num_letters=5, num_steps=8):
        # initialize network and optimizers
        self.loss = nn.CrossEntropyLoss()
        self.num_steps = num_steps
        self.iters = 0

        self.f = ptu.create_network(board_size, num_letters*3, n_layer, size, nn.ReLU(), nn.Identity())
        self.optimizer = optim.Adam(self.f.parameters(), lr=0.0003)

        pass

    def compute_bonus(self, boards: list, action: str, guess_count: list):
        s = 0
        for i in range(len(boards)):
            s += self._compute_single_board_bonus(boards[i], action, guess_count[i])
        return s

    def _compute_single_board_bonus(self, board: dict, action: str, guess_count: int):
        if self.iters % self.num_steps == 0:
            self.optimizer.zero_grad()
        # get score
        score = board["color"][guess_count]
        score = ptu.from_numpy(score)
        score = Fun.one_hot(score, num_classes=3).flatten()

        # one-hot encode the score

        # feed board into network
        state = board["board"]
        pred = self.f(state)
        loss = nn.CrossEntropyLoss(score, pred)
        loss.backward()
        self.iters += 1

        if self.iters % self.num_steps == self.num_steps - 1:
            self.optimizer.step()

        return loss