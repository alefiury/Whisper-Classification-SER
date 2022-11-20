import torch
from torch import nn
import pytorch_lightning as pl

from torchmetrics import Accuracy, MetricCollection

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)

from models.whisper_models import (
    MLPNetWhisper,
    CNN1DNetWhisper,
    CNN2DNetWhisper,
    WhisperForSpeechClassification
)

from models.basic_models import (
    MLPNet,
    CNN1DNet,
    CNN2DNet
)


class PlModelWrapper(pl.LightningModule):
    def __init__(
        self,
        config: dict = None
    ):
        super().__init__()
        self.save_hyperparameters()

        if config.training.use_pre_trained_data:
            if config.training.model_architecture == "mlp":
                self.model = MLPNet(**config.model)
            elif config.training.model_architecture == "cnn1d":
                self.model = CNN1DNet(**config.model)
            elif config.training.model_architecture == "cnn2d":
                self.model = CNN2DNet(**config.model)
            else:
                raise Exception(
                    "The model hasn't been defined! Your model must be one of these: ['mlp', 'cnn1d', 'cnn2d']"
                )

        else:
            if config.training.model_architecture == "mlp":
                self.model = MLPNetWhisper(**config.model)
            elif config.training.model_architecture == "cnn1d":
                self.model = CNN1DNetWhisper(**config.model)
            elif config.training.model_architecture == "cnn2d":
                self.model = CNN2DNetWhisper(**config.model)
            elif config.training.model_architecture == "finetune":
                self.model = WhisperForSpeechClassification(**config.model)
            else:
                raise Exception(
                        "The model hasn't been defined! Your model must be one of these: ['mlp', 'cnn1d', 'cnn2d', 'finetune']"
                    )

        self.config = config

        if config.training.loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
        if config.training.loss_func == "bce":
            self.criterion = nn.BCEWithLogitsLoss()

        metric_collection = MetricCollection([
            Accuracy(),
            MulticlassPrecision(
                num_classes=config.model.output_size
            ),
            MulticlassRecall(
                num_classes=config.model.output_size
            ),
            MulticlassF1Score(
                num_classes=config.model.output_size
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
        if self.config.training.loss_func == "bce":
            train_metrics = self.train_metrics(out, torch.argmax(y, axis=1))
        else:
            train_metrics = self.train_metrics(out, y)
        self.log_dict(train_metrics, on_step=True, on_epoch=True)

        return train_loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.model(x)
        val_loss = self.criterion(out, y)

        self.log('val_loss', val_loss)
        if self.config.training.loss_func == "bce":
            val_metrics = self.valid_metrics(out, torch.argmax(y, axis=1))
        else:
            val_metrics = self.valid_metrics(out, y)
        self.log_dict(val_metrics, on_step=True, on_epoch=True)


    def forward(self, batch):
        out = self.model(batch)
        return out


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.model(x)
        if self.config.training.loss_func == "bce":
            print(batch_idx, y, torch.argmax(out, axis=1).cpu().detach().numpy())
        else:
            test_loss = self.criterion(out, y)

        self.log('test_loss', test_loss, on_epoch=True)
        testing_metrics = self.test_metrics(out, y)
        self.log_dict(testing_metrics, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=self.config.training.scheduler_patience,
            min_lr=1.0e-6,
            factor=0.9
        )

        self.trainer.logger.experiment.config["scheduler"] = scheduler.__class__.__name__
        self.trainer.logger.experiment.config["optimizer"] = optimizer.__class__.__name__

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }