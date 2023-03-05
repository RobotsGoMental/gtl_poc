import torch


def apply_mask(mask, model):
    if mask is not None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad *= torch.tensor(mask[name], device=param.device)


def create_mask(cousins, model):
    def learning_spread(model_layer, cousin_layer):
        ls = torch.mean((cousin_layer - model_layer) ** 2, 0)
        min_ls = torch.min(ls)
        max_ls = torch.max(ls)
        spread = max_ls - min_ls
        return (ls - min_ls) / spread

    mask = {}
    model_dict = model.state_dict()

    cousins_stack = {}
    for cousin_dict in cousins:
        for name, param in model.named_parameters():
            cousins_stack.setdefault(name, [])
            cousins_stack[name].append(cousin_dict[name])  # TODO: this should be done differently...

    for name, param in model.named_parameters():
        mask[name] = learning_spread(model_dict[name], torch.stack(cousins_stack[name]))

    return mask
