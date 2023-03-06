import torch
from torch.utils.data import DataLoader


def apply_mask(mask, model):
    if mask is not None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad *= torch.tensor(mask[name], device=param.device)


def create_mask(cousins, model):
    def get_layer_mask(model_layer, cousin_layer):
        learning_spread = torch.mean((cousin_layer - model_layer) ** 2, 0)
        min_ls = torch.min(learning_spread)
        max_ls = torch.max(learning_spread)
        return (learning_spread - min_ls) / (max_ls - min_ls)

    mask = {}
    model_dict = model.state_dict()

    cousins_stack = {}
    for cousin_dict in cousins:
        for name, param in model.named_parameters():
            cousins_stack.setdefault(name, [])
            cousins_stack[name].append(cousin_dict[name])  # TODO: this should be done differently...

    for name, param in model.named_parameters():
        mask[name] = get_layer_mask(model_dict[name], torch.stack(cousins_stack[name]))

    return mask


def _create_cousin(model, data):
    cousin = model.clone()
    cousin.train()
    cousin_data = DataLoader(data, batch_size=32, shuffle=True, num_workers=4)
    for batch in cousin_data:
        x, y = batch
        loss = cousin(x, y)
        cousin.backward(loss)
        cousin.step()
        cousin.zero_grad()
    return cousin


def create_cousins(model, data):
    cousins = []
    for cousin_data in data:
        cousin = _create_cousin(model, cousin_data)
        cousins.append(cousin.model.state_dict())
    return cousins
