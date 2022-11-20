import os
import glob

import tqdm
import pandas as pd
from transformers import WhisperModel, AutoFeatureExtractor, Wav2Vec2Model

from preload_data import audio_to_embeddings_save_torch_file

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
    ravdess_base_dir = "../../data/Multidataset-ser"
    encoder = "whisper"

    audio_paths = glob.glob(
        os.path.join(
            ravdess_base_dir,
            '**',
            '*.wav'
        ),
        recursive=True
    )

    # print(len(audio_paths), audio_paths[:10])

    if encoder == "whisper":
        print("Using Whisper Embeddings... ")
        processor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        model = WhisperModel.from_pretrained("openai/whisper-base")
        model.freeze_encoder()
        model = model.get_encoder()

    if encoder == "wav2vec2":
        print("Using Wav2vec2 Embeddings... ")
        processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        model.eval()


    for filepath in tqdm.tqdm(audio_paths):
        audio_to_embeddings_save_torch_file(
            file_path=filepath,
            target_sampling_rate=16000,
            processor=processor,
            encoder=model,
            mean_pooled=False,
            base_dir="Multidataset-ser"
        )


if __name__ == "__main__":
    main()
