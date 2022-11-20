import os
import argparse
from collections import Counter

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from transformers import WhisperProcessor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from models.model_wrapper import PlModelWrapper
from utils.dataloader import DataGeneratorPreLoaded
from utils.evaluate import test_model
from utils.utils import convert_labels, save_conf_matrix, convert_labels_coraa_ser, convert_metadata_to_preloaded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("../", "config", "default_whisper.yaml"),
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
    parser.add_argument(
        "-m",
        "--metadata",
        default=False,
        action="store_true",
        help="Tests model"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    pl.seed_everything(42)

    if args.train:
        pl_model = PlModelWrapper(config)
        print(pl_model)

        wandb_logger = WandbLogger(
            name=os.path.basename(config.model_checkpoint.dirpath),
            project=config.logger.project_name,
            mode="disabled" if config.logger.debug else None
        )

        X_train = pd.read_csv(config.data.train_metadata_path)

        if config.data.val_metadata_path is None:
            X_train, X_val = train_test_split(
                X_train,
                test_size=0.2,
                random_state=42,
                stratify=X_train[[config.data.label_column]]
            )
        else:
            X_val = pd.read_csv(config.data.val_metadata_path)
        X_test = pd.read_csv(config.data.test_metadata_path)

        X_train = convert_labels(X_train, config.data.label_column)
        X_val = convert_labels(X_val, config.data.label_column)
        X_test = convert_labels(X_test, config.data.label_column)

        # print(X_train[config.data.label_column].value_counts())

        # X_train = convert_metadata_to_preloaded(X_train, config.data.filename_column, "whisper", "ravdess")
        # X_val = convert_metadata_to_preloaded(X_val, config.data.filename_column, "whisper", "ravdess")
        # X_test = convert_metadata_to_preloaded(X_test, config.data.filename_column, "whisper", "ravdess")

        train_dataset = DataGeneratorPreLoaded(
            dataset=X_train,
            label_column=config.data.label_column,
            filename_column=config.data.filename_column,
            base_dir=config.data.base_dir_data,
            use_mixup=config.data.use_mixup,
            mixup_alpha=config.data.mixup_alpha,
            data_type="train",
            use_hot_one_encoding=config.data.use_hot_one_encoding,
            use_add_noise=config.data.use_add_noise,
            min_amplitude=config.data.min_amplitude,
            max_amplitude=config.data.max_amplitude,
            class_num=config.model.output_size
        )

        val_dataset = DataGeneratorPreLoaded(
            dataset=X_val,
            label_column=config.data.label_column,
            filename_column=config.data.filename_column,
            base_dir=config.data.base_dir_data,
            use_mixup=False,
            mixup_alpha=None,
            data_type="val",
            use_hot_one_encoding=config.data.use_hot_one_encoding,
            use_add_noise=False,
            min_amplitude=None,
            max_amplitude=None,
            class_num=config.model.output_size
        )

        test_dataset = DataGeneratorPreLoaded(
            dataset=X_test,
            label_column=config.data.label_column,
            filename_column=config.data.filename_column,
            base_dir=config.data.base_dir_data,
            use_mixup=False,
            mixup_alpha=None,
            data_type="test",
            use_hot_one_encoding=config.data.use_hot_one_encoding,
            use_add_noise=False,
            min_amplitude=None,
            max_amplitude=None,
            class_num=config.model.output_size
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

        checkpoint_callback = EarlyStopping(**config.early_stopping)
        trainer = pl.Trainer(
            **config.trainer,
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(**config.model_checkpoint),
                checkpoint_callback,
                LearningRateMonitor("step")
            ],
            deterministic=True
        )
        trainer.fit(pl_model, train_loader, val_loader)
        trainer.test(pl_model, dataloaders=test_loader, ckpt_path="best", verbose=True)

    elif args.test:
        classes=[
            "neutral",
            "happiness",
            "sadness",
            "anger",
            "fear",
            "disgust",
            "surprise"
        ]

        # classes=[
        #     "neutral",
        #     "calm",
        #     "happy",
        #     "sad",
        #     "angry",
        #     "fearful",
        #     "disgust",
        #     "surprised"
        # ]

        # english, emovo
        # classes = [
        #     "neutral",
        #     "happy",
        #     "sad",
        #     "angry",
        #     "fear",
        #     "disgust",
        #     "surprise",
        # ]

        # # aesdd
        # classes = [
        #     "sad",
        #     "disgust"
        # ]

        # emodb
        # classes = [
        #     "neutral",
        #     "happy",
        #     "sad",
        #     "angry",
        #     "fear",
        #     "disgust"
        # ]

        # urdu
        # classes = [
        #     "neutral",
        #     "happy",
        #     "sad",
        #     "angry"
        # ]

        X_test = pd.read_csv(config.data.test_metadata_path)
        # X_test = convert_labels(X_test, config.data.label_column)
        X_test = convert_labels_coraa_ser(X_test, config.data.label_column)
        # processor = WhisperProcessor.from_pretrained(config.encoder_version)

        test_dataset = DataGeneratorPreLoaded(
            dataset=X_test,
            label_column=config.data.label_column,
            filename_column=config.data.filename_column,
            base_dir=config.data.base_dir_data,
            use_mixup=False,
            mixup_alpha=None,
            data_type="test",
            use_hot_one_encoding=config.data.use_hot_one_encoding,
            use_add_noise=False,
            min_amplitude=None,
            max_amplitude=None,
            class_num=config.model.output_size
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.training.num_workers
        )

        labels, pred_list = test_model(
            test_dataloader=test_loader,
            config=config,
            checkpoint_path="../checkpoints/whisper_cnn1d_preloaded_multilingual-True_mixup$-0.2_mixup_alpha-True_add_noise-300_epochs-bce_loss-[1024]_layers-[[32, 3, 1], [64, 1, 1]]_conv_layers-flat_pooling/epoch=287-step=53568.ckpt"
        )

        print(labels, pred_list)
        print(f"labels: {Counter(labels)} | pred: {Counter(pred_list)}")

        save_conf_matrix(
            targets=labels,
            preds=pred_list,
            classes=classes,
            output_path="../imgs/whisper_cnn1d_preloaded_multilingual-True_mixup$-0.2_mixup_alpha-True_add_noise-300_epochs-bce_loss-[1024]_layers-[[32, 3, 1], [64, 1, 1]]_conv_layers-flat_pooling_coraa.png"
        )

        rand_baseline = np.random.randint(8, size=len(labels))

        print(accuracy_score(labels, rand_baseline))
        print(f1_score(labels, rand_baseline, average="macro"))

        print(f"ACC: {accuracy_score(labels, pred_list)}")
        print(f"F1: {f1_score(labels, pred_list, average='macro')}")
        print(f"Precision: {precision_score(labels, pred_list, average='macro')}")
        print(f"Recall: {recall_score(labels, pred_list, average='macro')}")

if __name__ == "__main__":
    main()