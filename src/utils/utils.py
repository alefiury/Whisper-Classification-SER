from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix
from utils.dataloader import DataGeneratorPreLoaded


def save_conf_matrix(
    targets: List[int],
    preds: List[int],
    classes: List[str],
    output_path: str
) -> None:
    """
    Saves a confusion matrix given the true labels and the predicted outputs.
    """
    cm = confusion_matrix(
        y_true=targets,
        y_pred=preds
    )

    df_cm = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    plt.figure(figsize=(24,12))
    plot = sns.heatmap(df_cm, annot=True,  fmt='g')
    figure1 = plot.get_figure()
    plot.set_ylabel('True Label')
    plot.set_xlabel('Predicted Label')
    plt.tight_layout()
    figure1.savefig(output_path, format='png')


def load_preloaded_data(config):
    if config.train_preloaded_path is None:
        return None, None, None

    preloaded_train_dataset = load_from_disk(config.train_preloaded_path)
    preloaded_val_dataset = load_from_disk(config.val_preloaded_path)
    preloaded_test_dataset = load_from_disk(config.test_preloaded_path)

    preloaded_train_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )
    preloaded_val_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )
    preloaded_test_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )


    return preloaded_train_dataset, preloaded_val_dataset, preloaded_test_dataset

def convert_labels(df, label_column):
    emotion2int = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    df = df.replace({label_column: emotion2int})

    return df


def convert_labels_coraa_ser(df, label_column):
    emotion2int = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    multiple_labels = {
        "happiness/anger": "happiness",
        "*neutral": "neutral",
        "happiness/surprise": "happiness",
        "sadness/happiness": "sadness",
        "happiness/fear": "happiness",
        "surprise/happiness": "surprise",
        "happiness/sadness": "happiness",
        "*anger": "anger"
    }

    df = df.replace({label_column: multiple_labels})
    print(df[label_column].value_counts())
    df = df.replace({label_column: emotion2int})
    print(df[label_column].value_counts())

    return df


def convert_metadata_to_preloaded(df, file_path_column, sufix, base_dir):
    df[file_path_column] = df[file_path_column].str.replace(base_dir, base_dir+f"_{sufix}")
    df[file_path_column] = df[file_path_column].str.replace(".wav", ".pt")

    return df