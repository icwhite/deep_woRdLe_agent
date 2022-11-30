import torch
from torch import nn
import utils.infrastructure.pytorch_utils as ptu

class MLPPolicy(nn.Module): 

    def __init__(self, input_size: int, output_size: int, n_layers: int, size: int, activation, output_activation): 
        super().__init__()
        self.net = ptu.create_network(input_size, output_size, n_layers, size, activation, output_activation)

    def forward(self, obs): 
        obs = obs.to(torch.float32)
        return self.net(obs)

    def get_action(self, obs): 

        # Convert obs to tensor and call forward
        obs_t = ptu.from_numpy(obs)
        obs_t = obs_t.to(torch.float64)
        q_values = self(obs_t)
        
        # Get the action that has the highest q value
        action = torch.argmax(q_values)

        # Detach and return as integer
        action = action.detach().item()

        return action