import os

import torch
import torchaudio
import pandas as pd
from datasets import Dataset

def audio_file_to_array(
    batch,
    target_sampling_rate,
    processor,
    whiper_encoder,
    filename_column,
    base_dir_data
):
    waveform, source_sr = torchaudio.load(
        os.path.join(
            base_dir_data,
            batch[filename_column]
        )
    )

    # Convert to mono channel
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Convert source sampling rate to a target sampling rate
    if source_sr != target_sampling_rate:
        transform = torchaudio.transforms.Resample(source_sr, target_sampling_rate)
        waveform = transform(waveform)

    waveform = waveform.squeeze(0)

    inputs = processor(waveform, sampling_rate=target_sampling_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = whiper_encoder(**inputs)

    # Saves the 2D representation, shape: (1500, 512)
    # z = outputs.last_hidden_state.squeeze()

    # Saves the pooled representation, shape: (512)
    z = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    batch["embedding"] = z

    return batch


def prepare_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    target_sampling_rate: int,
    base_dir_output: str,
    whiper_encoder,
    processor,
    filename_column: str,
    base_dir_data: str
):
    train_path = os.path.join(base_dir_output, 'train')
    val_path = os.path.join(base_dir_output, 'val')
    test_path = os.path.join(base_dir_output, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_dataset = Dataset.from_pandas(X_train)
    val_dataset = Dataset.from_pandas(X_val)
    test_dataset = Dataset.from_pandas(X_test)

    print('Loading Audios... ')

    train_dataset = train_dataset.map(
        audio_file_to_array,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "whiper_encoder": whiper_encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data
        },
        num_proc=1
    )

    print(train_dataset)

    val_dataset = val_dataset.map(
        audio_file_to_array,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "whiper_encoder": whiper_encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data
        },
        num_proc=1
    )

    test_dataset = test_dataset.map(
        audio_file_to_array,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "whiper_encoder": whiper_encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data
        },
        num_proc=1
    )

    print('Saving Dataset... ')

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    test_dataset.save_to_disk(test_path)