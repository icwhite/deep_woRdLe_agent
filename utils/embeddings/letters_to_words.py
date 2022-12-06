import pytorch_lightning as pl
from torch import nn
import torch
import utils.infrastructure.pytorch_utils as ptu
import torch.nn.functional as Fun

class LetterToWord(nn.Module):
    def __init__(self, n_letters, n_possible_words, n_layers, size):

        self.f = ptu.create_network(26*n_letters, n_possible_words, n_layers, size, nn.ReLU(), nn.Identity())
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.f.parameters(), lr=1e-4)

    def forward(self, letters):
        return self.f(letters)

    def update(self, letters, word):

        pred = self(letters)
        loss = self.loss(pred, word)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Training Loss": loss
        }

