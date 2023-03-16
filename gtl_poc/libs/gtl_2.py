import torch
from torch.nn import Parameter, Module


def _get_guidance_matrix(scout_param, model_param: Parameter) -> torch.Tensor:
    learning_spread = torch.mean((scout_param - model_param) ** 2, 0)
    min_ls = torch.min(learning_spread)
    max_ls = torch.max(learning_spread)
    return torch.tensor((learning_spread - min_ls) / (max_ls - min_ls))


def apply_guidance_matrices(model: Module, scouts) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad *= _get_guidance_matrix(scouts[name], param)


def debias_layers(model: Module) -> None:
    def debias(param):
        return torch.stack([torch.mean(param, dim=0)] * len(param), dim=0)

    last_layer = list(model.children())[-1]
    last_layer.bias = debias(last_layer.bias)
    last_layer.weight = debias(last_layer.weight)
