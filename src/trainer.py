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

from models.model_wrapper import PlModelWrapper
from utils.dataloader import DataGeneratorForWhisper
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
            project=config.logger.project_name,
            mode="disabled" if config.logger.debug else None
        )

        X_train = pd.read_csv(config.data.train_metadata_path)
        X_val = pd.read_csv(config.data.val_metadata_path)
        X_test = pd.read_csv(config.data.test_metadata_path)

        processor = WhisperProcessor.from_pretrained(config.encoder_version)

        train_dataset = DataGeneratorForWhisper(
            df=X_train,
            processor=processor,
            label_column=config.data.label_column,
            base_wav_path=config.data.base_dir_data,
            filename_column=config.data.filename_column,
            target_sampling_rate=config.data.target_sampling_rate,
            use_mixup=config.data.use_mixup,
            mixup_alpha=config.data.mixup_alpha,
            use_specaug=config.data.use_specaug,
            specaug_freqm=config.data.specaug_freqm,
            specaug_timem=config.data.specaug_timem,
            class_num=config.model.output_size,
            data_type="train"
        )

        val_dataset = DataGeneratorForWhisper(
            df=X_val,
            target_sampling_rate=config.data.target_sampling_rate,
            base_wav_path=config.data.base_dir_data,
            processor=processor,
            filename_column=config.data.filename_column,
            label_column=config.data.label_column,
            use_mixup=False,
            mixup_alpha=0.0,
            use_specaug=False,
            specaug_freqm=0.0,
            specaug_timem=0.0,
            class_num=config.model.output_size,
            data_type="val"
        )

        test_dataset = DataGeneratorForWhisper(
            df=X_test,
            target_sampling_rate=config.data.target_sampling_rate,
            base_wav_path=config.data.base_dir_data,
            processor=processor,
            filename_column=config.data.filename_column,
            label_column=config.data.label_column,
            use_mixup=False,
            mixup_alpha=0.0,
            use_specaug=False,
            specaug_freqm=0.0,
            specaug_timem=0.0,
            class_num=config.model.output_size,
            data_type="test"
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
        print(f"Best Checkpoint: {checkpoint_callback.best_model_path}")

    elif args.test:
        X_test = pd.read_csv(config.data.test_metadata_path)

        test_dataset = DataGeneratorForWhisper(
            df=X_test,
            target_sampling_rate=config.data.target_sampling_rate,
            base_wav_path=config.data.base_dir_data,
            processor=processor,
            filename_column=config.data.filename_column,
            label_column=config.data.label_column,
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