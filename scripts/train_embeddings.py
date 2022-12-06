import pytorch_lightning as pl
from utils.embeddings.vae import VAE

wordle_words = open("scripts/wordle_words.txt", "r").read().split(",")
wordle_words = [word.replace('\n', '') for word in wordle_words]


vae = VAE()
# trainer = pl.Trainer(gpus=1, max_epochs=20, callbacks=[sampler])
# trainer.fit(vae, dataset)