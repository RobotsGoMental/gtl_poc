import os

import torch
from pytorch_lightning import seed_everything

from libs.MaskedModel import MaskedModel
from libs.RandomBinaryData import RandomBinaryData
from libs.functions import train_multilayer
from libs.gtl import create_mask

checkpoint_dir_name = 'checkpoints'

categories = 10
inputs = 25
examples = 200
num_cousins = 15
cousin_size = 4

seed_everything(347263, workers=True)
binary = RandomBinaryData(inputs, categories, num_cousins, examples, cousin_size)

if not os.path.exists(f'{checkpoint_dir_name}/tl_base.ckpt'):
    multilayer = train_multilayer(data=binary.data(), num_categories=categories, name='multilayer_10_categ_tl_base')
    multilayer.save_checkpoint(f'{checkpoint_dir_name}/tl_base.ckpt')

all_cousins = []
for i in range(num_cousins):
    print(f"cousins {i}")
    cousin = train_multilayer(data=binary.data(i), name='cousins', checkpoint='tl_base', convergence=False)
    all_cousins.append(cousin.model.state_dict())
mask = create_mask(cousins=all_cousins, model=MaskedModel.load_from_checkpoint(f'{checkpoint_dir_name}/tl_base.ckpt'))
torch.save(mask, f'{checkpoint_dir_name}/mask.pt')

for i in range(num_cousins):
    data = binary.data(i)
    print(f"situation_tl {i}")
    situation = train_multilayer(data=data, name='situation_tl', checkpoint='tl_base', convergence=False)
    print(f"situation_tl_masked {i}")
    situation_masked = train_multilayer(data=data, name='situation_tl_masked', checkpoint='tl_base', mask=mask,
                                        convergence=False)
