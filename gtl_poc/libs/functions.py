import logging
import warnings

from pytorch_lightning import Trainer
import torch
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


def checkpoint_exists(name):
    return os.path.exists(f'{checkpoint_dir_name}/{name}.ckpt')


def save_checkpoint(model, name):
    model.save_checkpoint(f'{checkpoint_dir_name}/{name}.ckpt')


def get_mask():
    return torch.load(f'{checkpoint_dir_name}/mask.pt')


def train_multilayer(data, num_categories=10, name='multilayer', mask=None, checkpoint=None,
                     convergence=True):
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


def create_mask(cousins, model_checkpoint):
    def _compute_mask(model, cousin_dict_list):
        def learning_spread(model, cousin):
            ls = torch.mean((cousin - model) ** 2, 0)
            min_ls = torch.min(ls)
            max_ls = torch.max(ls)
            spread = max_ls - min_ls
            return (ls - min_ls) / spread

        out_mask = {}
        model_dict = model.state_dict()

        cousins = {}
        for cousin_dict in cousin_dict_list:
            for name, param in model.named_parameters():
                cousins.setdefault(name, [])
                cousins[name].append(cousin_dict[name])  # TODO: this should be done differently...

        for name, param in model.named_parameters():
            out_mask[name] = learning_spread(model_dict[name], torch.stack(cousins[name]))

        return out_mask

    model = MaskedModel.load_from_checkpoint(f'{checkpoint_dir_name}/{model_checkpoint}.ckpt')
    mask = _compute_mask(model, cousins)
    torch.save(mask, f'{checkpoint_dir_name}/mask.pt')
    return mask
