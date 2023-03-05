import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class MultilayerLightningModel(LightningModule):
    def __init__(self, inputs, hidden, last_layer, linear_layers=1):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(inputs, hidden)
        self.linear_h = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(linear_layers - 1)])
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden, last_layer)

    def forward(self, x):
        x = self.Sigmoid(self.linear(x))
        for layer in self.linear_h:
            x = self.Sigmoid(layer(x))
        return self.linear2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        correct = sum(a.argmax() == b.argmax() for a, b in zip(y_hat, y))
        total = y.shape[0]
        return {"loss": loss, "correct": correct, "total": total}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = sum(x["correct"] for x in outputs) / sum(x["total"] for x in outputs)
        self.log("Loss/Epoch", avg_loss, on_epoch=True, prog_bar=True)
        self.log("Acc/Epoch", avg_acc, on_epoch=True, prog_bar=True)
