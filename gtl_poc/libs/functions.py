import logging
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

import os

from libs.MaskedModel import MaskedModel

checkpoint_dir_name = 'checkpoints'
logs_dir = 'logs'

num_workers = 1
log_every_n_steps = 100
num_epochs = 1400
batch_size = 200
num_layers = 4
num_hidden = 80
num_inputs = 25

stopping_threshold = 0.80

count = 0
log_dir_name = f'{logs_dir}/{count}'
# checkpoint_dir_name = f'{checkpoint_dir}/{count}'
while os.path.exists(log_dir_name):
    log_dir_name = f'{logs_dir}/{count}'
    # checkpoint_dir_name = f'{checkpoint_dir}/{count}'
    count += 1

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def train_multilayer(data, num_categories=10, name='multilayer', mask=None, checkpoint=None, convergence=True):
    tensorboard = pl_loggers.TensorBoardLogger(save_dir='logs', name=name)

    if checkpoint:
        model = MaskedModel.load_from_checkpoint(f"{checkpoint_dir_name}/{checkpoint}.ckpt", mask=mask)
    else:
        model = MaskedModel(mask, num_inputs, num_hidden, num_categories, num_layers)

    train_loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)
    early_stop_callback = EarlyStopping(monitor="Acc/Epoch", min_delta=0.00, patience=num_epochs, verbose=True,
                                        mode="max", stopping_threshold=stopping_threshold)
    callbacks = [early_stop_callback] if convergence else []
    trainer = Trainer(enable_model_summary=False, max_epochs=num_epochs, logger=tensorboard,
                      callbacks=callbacks, log_every_n_steps=log_every_n_steps)
    trainer.fit(model, train_loader)
    return trainer
