from libs.MultilayerLightningModel import MultilayerLightningModel
from libs.gtl import apply_mask


class MaskedModel(MultilayerLightningModel):
    def __init__(self, mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def on_after_backward(self):
        apply_mask(mask=self.mask, model=self)
