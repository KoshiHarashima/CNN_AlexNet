# train/train_lightning.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from resnet import resnet50

class LitResNet(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3, use_se=False, dropout_rate=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50(num_classes=num_classes, use_se=use_se, dropout_rate=dropout_rate)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
