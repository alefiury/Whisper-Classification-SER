import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score
)

from models.model import MLPNet


class MLPNetWrapper(pl.LightningModule):
    def __init__(self, config: dict = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPNet(**config.model)
        self.config = config

        self.criterion = nn.CrossEntropyLoss()

        metric_collection = MetricCollection([
            Accuracy(),
            Precision(
                num_classes=config.model.output_size,
                average='macro'
            ),
            Recall(
                num_classes=config.model.output_size,
                average='macro'
            ),
            F1Score(
                num_classes=config.model.output_size,
                average='macro'
            )
        ])

        self.train_metrics = metric_collection.clone(prefix='train_')
        self.valid_metrics = metric_collection.clone(prefix='val_')
        self.test_metrics = metric_collection.clone(prefix='test_')


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.model(x)
        train_loss = self.criterion(out, y)

        self.log('train_loss', train_loss)
        train_metrics = self.train_metrics(out, y)
        self.log_dict(train_metrics, on_step=True, on_epoch=True)

        return train_loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.model(x)
        val_loss = self.criterion(out, y)

        self.log('val_loss', val_loss)
        val_metrics = self.valid_metrics(out, y)
        self.log_dict(val_metrics, on_step=True, on_epoch=True)


    def forward(self, batch):
        out = self.model(batch)
        return out


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.model(x)
        test_loss = self.criterion(out, y)

        self.log('test_loss', test_loss, on_epoch=True)
        testing_metrics = self.test_metrics(out, y)
        self.log_dict(testing_metrics, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.lr
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=self.config.training.scheduler_patience,
            factor=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step"
            }
        }