import os
from pathlib import Path

from downscaling.autoencoder.autoencoder import AutoEncoder, WeightedVectorLoss

checkpoint_path = Path(os.getenv('CHECKPOINT_ROOT', './checkpoints')) / 'autoencoder'
_weights_path = checkpoint_path.joinpath('20220708_0737/weights.ckpt/variables/variables').resolve()


def build_autoencoder():
    x = AutoEncoder(latent_dimension=96,
                    time_steps=24, img_size=96, batch_size=8)
    x.compile('adam', loss=WeightedVectorLoss())
    return x


autoencoder = build_autoencoder()
autoencoder.load_weights(_weights_path)
encoder = autoencoder.encoder
