import pytorch_lightning as pl
pl.seed_everything(1234)
from torch import nn
import torch
import utils.infrastructure.pytorch_utils as ptu
# from pl_bolts.models.autoencoders.components import (
#     resnet18_decoder,
#     resnet18_encoder,
# )

# code from this Github: https://github.com/williamFalcon/pytorch-lightning-vae


class VAE(pl.LightningModule):
    def __init__(self, input_dim, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = ptu.create_network(input_dim, enc_out_dim, 2, 128, nn.ReLU(), nn.Identity())
        self.decoder = ptu.create_network(enc_out_dim, input_dim, 2, 128, nn.ReLU(), nn.Identity())

        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)

        recon_loss = nn.MSELoss(x, x_decoded)

        self.log_dict({
            'reconstruction': recon_loss.mean(),
        })

        return recon_loss
