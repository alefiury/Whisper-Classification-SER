import os
import glob

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor, Wav2Vec2Model

from preload_data import prepare_data

def create_metadata_ravdess(ravdess_base_dir: str):
    """
    Creates a metadata of the ravdess dataset
    """
    int2emotion = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    audio_paths = glob.glob(
        os.path.join(
            ravdess_base_dir,
            '**',
            '*.wav'
        ),
        recursive=True
    )

    # print(audio_paths)

    genders = []
    emotions = []
    actors = []
    labels = []

    for _, audio_path in enumerate(audio_paths):
        filename = os.path.basename(audio_path).split('.')[0]

        modality = filename.split('-')[0]
        vocal_channel = filename.split('-')[1]
        emotion = filename.split('-')[2]
        actor = filename.split('-')[6]

        assert modality=="03"
        assert vocal_channel=="01"

        if int(actor)%2 == 0:
            gender = 'female'
        else:
            gender = 'male'


        genders.append(gender)
        emotions.append(int2emotion[emotion])
        actors.append(int(actor))
        labels.append(int(emotion)-1)

    df = pd.DataFrame(
        list(
            zip(
                audio_paths,
                genders,
                emotions,
                actors,
                labels
            )
        ),
        columns=[
            'wav_file',
            'gender',
            'emotion',
            'actor',
            'label'
        ]
    )

    return df


def split_dataset(
    df: pd.DataFrame = None,
    frac_train: float = 0.70,
    frac_val: float = 0.10,
    frac_test: float = 0.20,
):
    """
    Split the data into subsets for training, validation, and testing.
    The same speakers are present in all subsets.
    """
    X_train, X_temp, _, _ = train_test_split(
        df,
        df["label"],
        test_size=(1.0 - frac_train),
        stratify=df[["gender", "emotion", "actor"]],
        random_state=42
    )

    X_val, X_test, _, _  = train_test_split(
        X_temp,
        X_temp["label"],
        test_size=frac_test,
        stratify=X_temp[["gender", "emotion", "actor"]],
        random_state=42
    )

    return X_train, X_val, X_test


def split_dataset_actors(df: pd.DataFrame):
    """
    Split the data into subsets for training, validation, and testing.
    The speakers are split up so that there is a distinct speakers for each subset.
    """
    train_actors = [i for i in range(1, 17)]
    val_actors = [i for i in range(17, 19)]
    test_actors = [i for i in range(19, 25)]

    X_train = df[df["actor"].isin(train_actors)]
    X_val = df[df["actor"].isin(val_actors)]
    X_test = df[df["actor"].isin(test_actors)]

    return X_train, X_val, X_test


def main():
    ravdess_df = create_metadata_ravdess(
        ravdess_base_dir="../../data/ravdess"
    )
    encoder = "whisper"
    use_pooling = True
    base_dir_output = f"../../data/ravdess_preloaded_{encoder}_pooled"

    X_train, X_val, X_test = split_dataset_actors(df=ravdess_df)

    X_train.to_csv("../../data/ravdess_train_metadata.csv", index=False)
    X_val.to_csv("../../data/ravdess_eval_metadata.csv", index=False)
    X_test.to_csv("../../data/ravdess_test_metadata.csv", index=False)

    # if encoder == "whisper":
    #     print("Using Whisper Embeddings... ")
    #     processor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    #     model = WhisperModel.from_pretrained("openai/whisper-base")
    #     model = model.encoder
    #     model.eval()

    # if encoder == "wav2vec2":
    #     print("Using Wav2vec2 Embeddings... ")
    #     processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    #     model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
    #     model.eval()

    # prepare_data(
    #     X_train=X_train,
    #     X_val=X_val,
    #     X_test=X_test,
    #     target_sampling_rate=16000,
    #     base_dir_output=base_dir_output,
    #     encoder=model,
    #     processor=processor,
    #     filename_column="wav_file",
    #     base_dir_data="",
    #     mean_pooled=use_pooling
    # )


if __name__ == "__main__":
    main()
