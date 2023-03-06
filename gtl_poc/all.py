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
else:
    multilayer = MaskedModel.load_from_checkpoint(f"{checkpoint_dir_name}/tl_base.ckpt")

all_cousins = []
for i in range(num_cousins):
    print(f"cousins {i}")
    cousin = train_multilayer(data=binary.data(i), name='cousins', checkpoint='tl_base', convergence=False)
    all_cousins.append(cousin.model.state_dict())
mask = create_mask(cousins=all_cousins, model=multilayer)
torch.save(mask, f'{checkpoint_dir_name}/mask.pt')

for i in range(num_cousins):
    data = binary.data(i)
    print(f"situation_tl {i}")
    situation = train_multilayer(data=data, name='situation_tl', checkpoint='tl_base', convergence=False)
    print(f"situation_tl_masked {i}")
    situation_masked = train_multilayer(data=data, name='situation_tl_masked', checkpoint='tl_base', mask=mask,
                                        convergence=False)
# rewrite - how to add a function after loss.backward() and before optimizers are stepped in pytorch?
# In PyTorch, you can add a function after loss.backward() and before the optimizer's step() method by defining a closure that wraps the backward pass and the optimization step. You can then add your additional function between the backward pass and the optimization step in the closure.
#
# Here's an example of how to add a function after loss.backward() and before the optimizer's step() method:
#
# python
#
# import torch
# import torch.optim as optim
#
# # Define the model, loss function, and optimizer
# model = MyModel()
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # Define a closure that wraps the backward pass and the optimization step
# def closure():
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)
#     loss.backward()
#     # Call your additional function here
#     additional_function()
#     return loss
#
# # Perform a training step
# inputs = torch.randn(10, 3)
# targets = torch.LongTensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
# loss = optimizer.step(closure)
#
# # Define your additional function
# def additional_function():
#     with torch.no_grad():
#         # Perform some additional operation on the model's parameters
#         for param in model.parameters():
#             param.add_(0.1 * param.grad)
#
# In this example, we define a model, loss function, and optimizer. We define a closure closure that wraps the backward pass and the optimization step. The closure calls optimizer.zero_grad() to clear the gradients, computes the forward pass of the model on inputs, calculates the loss with loss_fn, and calls loss.backward() to compute the gradients of the loss with respect to the model's parameters.
#
# We then call our additional function additional_function() between the backward pass and the optimization step in the closure. Finally, we return the loss from the closure.
#
# We perform a training step by calling optimizer.step(closure) with the closure as its argument. The optimizer will call the closure to perform the backward pass and the optimization step.
#
# We define our additional function additional_function() outside of the closure. In this case, the function performs some additional operation on the model's parameters by adding 0.1 * param.grad to each parameter.