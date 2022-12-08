# Idea: For first few games, just want to learn an implicit model of how likely you are to get a certain score
# when you guess a word.
from utils.exploration.base_exploration_model import BaseExplorationModel
from torch import nn
import utils.infrastructure.pytorch_utils as ptu
import torch.optim as optim
import torch.nn.functional as Fun
import numpy as np
import torch

class ModelScoreExplore(BaseExplorationModel):

    def __init__(self, n_layer=2, size=128, board_size=30, num_letters=5, num_steps=8):

        self.num_letters = num_letters # number of letters per guess
        # initialize network and optimizers
        self.loss = nn.CrossEntropyLoss()
        self.num_steps = num_steps
        self.iters = 0

        # input will be one-hot encoded letters + one-hot encoded colors for input state

        self.f = ptu.create_network(board_size*27 + board_size*4, num_letters*3, n_layer, size, nn.ReLU(), nn.Identity())
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
        # get score this is training target
        score = board["colors"][guess_count - 1] # guess_count starts from 1
        score = ptu.from_numpy(score).to(torch.int64)
        score = Fun.one_hot(score, num_classes=3).flatten()
        score = score.to(torch.float32)

        # one-hot encode the score

        # feed board into network
        letters = board["letters"]
        letters = ptu.from_numpy(letters).to(torch.int64) + 1
        letters = Fun.one_hot(letters, num_classes=27).flatten()

        # zero out the scores before passing them to model
        colors = board["colors"]
        colors[guess_count - 1] = np.zeros(self.num_letters) - 1
        colors = ptu.from_numpy(colors).to(torch.int64) + 1
        colors = Fun.one_hot(colors, num_classes=4).flatten()

        # concatenate together to get full state vector minus next score
        state = torch.concat([letters, colors]).to(torch.float32)

        pred = self.f(state)
        loss = self.loss(score, pred)
        loss.backward()

        if self.iters % self.num_steps == self.num_steps - 1:
            self.optimizer.step()

        self.iters += 1
        return ptu.to_numpy(loss)