import os
import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from datasets import load_from_disk
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import WhisperProcessor, WhisperModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from models.model_wrapper import MLPNetWrapper
from utils.dataloader import DataGenerator, DataGeneratorPreLoaded
from utils.evaluate import test_model
from utils.utils import load_preloaded_data, save_conf_matrix

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
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    pl.seed_everything(42)

    if args.train:
        mlp_net = MLPNetWrapper(config)
        print(mlp_net)
        wandb_logger = WandbLogger(
            project=config.logger.project_name,
            mode="disabled" if config.logger.debug else None
        )

        if config.data.use_preloaded_data:
            preloaded_train, preloaded_val, preloaded_test = load_preloaded_data(config.data.preloaded_loading.dataset)
            preloaded_train2, preloaded_val2, preloaded_test2 = load_preloaded_data(config.data.preloaded_loading.dataset2)

            train_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_train,
                embedding_column=config.data.preloaded_loading.dataset.embedding_column,
                label_column=config.data.preloaded_loading.dataset.label_column,
                dataset2=preloaded_train2 if preloaded_train2 else None,
                file_path_column=config.data.preloaded_loading.dataset2.file_path_column if preloaded_train2 else None,
                concatanation_type=config.data.preloaded_loading.dataset2.concatanation_type if preloaded_train2 else None,
                add_gaussian_noise=False
            )

            val_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_val,
                embedding_column=config.data.preloaded_loading.dataset.embedding_column,
                label_column=config.data.preloaded_loading.dataset.label_column,
                dataset2=preloaded_val2 if preloaded_val2 else None,
                file_path_column=config.data.preloaded_loading.dataset2.file_path_column if preloaded_val2 else None,
                concatanation_type=config.data.preloaded_loading.dataset2.concatanation_type if preloaded_val2 else None
            )

            test_dataset = DataGeneratorPreLoaded(
                dataset=preloaded_test,
                embedding_column=config.data.preloaded_loading.dataset.embedding_column,
                label_column=config.data.preloaded_loading.dataset.label_column,
                dataset2=preloaded_test2 if preloaded_test2 else None,
                file_path_column=config.data.preloaded_loading.dataset2.file_path_column if preloaded_test2 else None,
                concatanation_type=config.data.preloaded_loading.dataset2.concatanation_type if preloaded_test2 else None
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
        # trainer.test(mlp_net, dataloaders=test_loader, ckpt_path="last", verbose=True)

    elif args.test:
        preloaded_test_dataset = load_from_disk(config.data.preloaded_loading.dataset.test_preloaded_path)
        _, _, preloaded_test2 = load_preloaded_data(config.data.preloaded_loading.dataset2)
        preloaded_test_dataset.set_format(
            type='torch',
            columns=[config.data.preloaded_loading.dataset.embedding_column, config.data.preloaded_loading.dataset.label_column]
        )

        test = DataGeneratorPreLoaded(
                dataset=preloaded_test_dataset,
                dataset2=preloaded_test2 if preloaded_test2 else None,
                file_path_column=config.data.preloaded_loading.dataset2.file_path_column if preloaded_test2 else None,
                concatanation_type=config.data.preloaded_loading.dataset2.concatanation_type if preloaded_test2 else None,
                embedding_column=config.data.preloaded_loading.dataset.embedding_column,
                label_column=config.data.preloaded_loading.dataset.label_column
            )
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
            checkpoint_path="../checkpoints/whisper_wav2vec2_classification_actor/epoch=88-step=5340.ckpt"
        )

        save_conf_matrix(
            targets=labels,
            preds=pred_list,
            classes=[
                "neutral",
                "calm",
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgust",
                "surprised"
            ],
            output_path="wav2vec2_conf_cnn1d.png"
        )

        rand_baseline = np.random.randint(8, size=len(labels))

        print(accuracy_score(labels, rand_baseline))
        print(f1_score(labels, rand_baseline, average="macro"))

        print(accuracy_score(labels, pred_list))
        print(f1_score(labels, pred_list, average="macro"))
        print(recall_score(labels, pred_list, average="macro"))
        print(precision_score(labels, pred_list, average="macro"))


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()