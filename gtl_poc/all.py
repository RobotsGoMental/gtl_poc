from pytorch_lightning import seed_everything

from libs.RandomBinaryData import RandomBinaryData
from libs.functions import train_multilayer, create_mask, save_checkpoint, checkpoint_exists

categories = 10
inputs = 25
examples = 200
num_cousins = 15
cousin_size = 4

seed_everything(347263, workers=True)
binary = RandomBinaryData(inputs, categories, num_cousins, examples, cousin_size)

if not checkpoint_exists('tl_base'):
    multilayer = train_multilayer(data=binary.data(), num_categories=categories, name='multilayer_10_categ_tl_base')
    save_checkpoint(multilayer, 'tl_base')

all_cousins = []
for i in range(num_cousins):
    print(f"cousins {i}")
    cousin = train_multilayer(data=binary.data(i), name='cousins', checkpoint='tl_base', convergence=False)
    all_cousins.append(cousin.model.state_dict())
mask = create_mask(cousins=all_cousins, model_checkpoint='tl_base')

for i in range(num_cousins):
    data = binary.data(i)
    print(f"situation_tl {i}")
    situation = train_multilayer(data=data, name='situation_tl', checkpoint='tl_base', convergence=False)
    print(f"situation_tl_masked {i}")
    situation_masked = train_multilayer(data=data, name='situation_tl_masked', checkpoint='tl_base', mask=mask,
                                        convergence=False)
