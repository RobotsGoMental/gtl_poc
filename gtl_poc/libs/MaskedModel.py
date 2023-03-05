import torch

from libs.MultilayerLightningModel import MultilayerLightningModel


class MaskedModel(MultilayerLightningModel):
    def __init__(self, mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def on_after_backward(self):
        if self.mask is not None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.grad *= torch.tensor(self.mask[name], device=param.device)
