import os
import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from datasets import load_from_disk
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, f1_score
from transformers import WhisperProcessor, WhisperModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from models.model_wrapper import MLPNetWrapper
from utils.dataloader import DataGenerator, DataGeneratorPreLoaded
from utils.evaluate import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("../", "config", "default.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-tr",
        "--train",
        default=False,
        action="store_true",
        help="Trains model"
    )
    parser.add_argument(
        "-te",
        "--test",
        default=False,
        action="store_true",
        help="Tests model"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    pl.seed_everything(42)

    if args.train:
        mlp_net = MLPNetWrapper(config)
        wandb_logger = WandbLogger(
            project=config.logger.project_name,
            mode="disabled" if config.logger.debug else None
        )

        if config.data.use_preloaded_data:
            preloaded_train_dataset = load_from_disk(config.data.train_preloaded_path)
            preloaded_val_dataset = load_from_disk(config.data.val_preloaded_path)
            preloaded_test_dataset = load_from_disk(config.data.test_preloaded_path)

            preloaded_train_dataset.set_format(
                type='torch',
                columns=[config.data.embedding_column, config.data.label_column]
            )
            preloaded_val_dataset.set_format(
                type='torch',
                columns=[config.data.embedding_column, config.data.label_column]
            )
            preloaded_test_dataset.set_format(
                type='torch',
                columns=[config.data.embedding_column, config.data.label_column]
            )

            train_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_train_dataset,
                embedding_column=config.data.embedding_column,
                label_column=config.data.label_column
            )
            val_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_val_dataset,
                embedding_column=config.data.embedding_column,
                label_column=config.data.label_column
            )
            test_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_test_dataset,
                embedding_column=config.data.embedding_column,
                label_column=config.data.label_column
            )

        else:
            X_train = pd.read_csv(config.data.metadata_path)
            X_val = pd.read_csv(config.data.metadata_path)
            X_test = pd.read_csv(config.data.metadata_path)

            processor = WhisperProcessor.from_pretrained(config.whisper.whisper_version)
            model = WhisperModel.from_pretrained(config.whisper.whisper_version)

            model.eval()

            train_dataset = DataGenerator(
                df=X_train,
                target_sampling_rate=config.data.target_sampling_rate,
                base_wav_path=config.data.base_dir_data,
                processor=processor,
                whiper_encoder=model.encoder,
                filename_column=config.data.filename_column,
                label_column=config.data.label_column
            )

            val_dataset = DataGenerator(
                df=X_val,
                target_sampling_rate=config.data.target_sampling_rate,
                base_wav_path=config.data.base_dir_data,
                processor=processor,
                whiper_encoder=model.encoder,
                filename_column=config.data.filename_column,
                label_column=config.data.label_column
            )

            test_dataset = DataGenerator(
                df=X_test,
                target_sampling_rate=config.data.target_sampling_rate,
                base_wav_path=config.data.base_dir_data,
                processor=processor,
                whiper_encoder=model.encoder,
                filename_column=config.data.filename_column,
                label_column=config.data.label_column
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.training.num_workers
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.training.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.training.num_workers
        )

        trainer = pl.Trainer(
            **config.trainer,
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(**config.model_checkpoint),
                EarlyStopping(**config.early_stopping),
                LearningRateMonitor("step")
            ],
            deterministic=True
        )
        trainer.fit(mlp_net, train_loader, val_loader)
        trainer.test(mlp_net, dataloaders=test_loader, ckpt_path="best", verbose=True)

    elif args.test:
        preloaded_test_dataset = load_from_disk(config.data.test_preloaded_path)
        preloaded_test_dataset.set_format(
            type='torch',
            columns=[config.data.embedding_column, config.data.label_column]
        )
        test = DataGenDatasets(preloaded_test_dataset)
        test_loader = torch.utils.data.DataLoader(
            test,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.training.num_workers
        )

        labels, pred_list = test_model(
            test_dataloader=test_loader,
            config=config,
            checkpoint_path="checkpoints/whisper_classification_actor/epoch=31-step=1920.ckpt"
        )

        # save_conf_matrix(
        #     targets=labels,
        #     preds=pred_list,
        #     classes=[
        #         "neutral",
        #         "calm",
        #         "happy",
        #         "sad",
        #         "angry",
        #         "fearful",
        #         "disgust",
        #         "surprised"
        #     ],
        #     output_path="whisper_conf.png"
        # )

        rand_baseline = np.random.randint(8, size=len(labels))

        print(accuracy_score(labels, rand_baseline))
        print(f1_score(labels, rand_baseline, average="macro"))


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()