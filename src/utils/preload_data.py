import os

import torch
import torchaudio
import pandas as pd
from datasets import Dataset


def audio_to_embeddings(
    batch,
    target_sampling_rate,
    processor,
    encoder,
    filename_column,
    base_dir_data,
    mean_pooled
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
        outputs = encoder(**inputs)

    if mean_pooled:
        # Saves the pooled representation, whisper shape: (512), wav2vec2 (xls-r 300m) shape: (1024)
        z = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    else:
        # Saves the 2D representation, whisper shape: (1500, 512), wav2vec2 (xls-r 300m) shape: (t, 1024)
        z = outputs.last_hidden_state.squeeze()

    batch["embedding"] = z

    return batch


def audio_to_embeddings_save_torch_file(
    file_path,
    target_sampling_rate,
    processor,
    encoder,
    mean_pooled,
    base_dir
):
    waveform, source_sr = torchaudio.load(file_path)

    # Convert to mono channel
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Convert source sampling rate to a target sampling rate
    if source_sr != target_sampling_rate:
        transform = torchaudio.transforms.Resample(source_sr, target_sampling_rate)
        waveform = transform(waveform)

    waveform = waveform.squeeze(0)

    inputs = processor(waveform, sampling_rate=target_sampling_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)

    if mean_pooled:
        # Saves the pooled representation, whisper shape: (512), wav2vec2 (xls-r 300m) shape: (1024)
        z = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    else:
        # Saves the 2D representation, whisper shape: (1500, 512), wav2vec2 (xls-r 300m) shape: (t, 1024)
        z = outputs.last_hidden_state.squeeze()

    new_path = file_path.replace(base_dir, base_dir+"_preloaded_2D").replace(".wav", ".pt")

    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    torch.save(z, new_path)


def prepare_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    target_sampling_rate: int,
    base_dir_output: str,
    encoder,
    processor,
    filename_column: str,
    base_dir_data: str,
    mean_pooled: bool
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
        audio_to_embeddings,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "encoder": encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data,
            "mean_pooled": mean_pooled
        },
        num_proc=1
    )

    print(train_dataset)

    val_dataset = val_dataset.map(
        audio_to_embeddings,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "encoder": encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data,
            "mean_pooled": mean_pooled
        },
        num_proc=1
    )

    test_dataset = test_dataset.map(
        audio_to_embeddings,
        fn_kwargs={
            "target_sampling_rate": target_sampling_rate,
            "encoder": encoder,
            "processor": processor,
            "filename_column": filename_column,
            "base_dir_data": base_dir_data,
            "mean_pooled": mean_pooled
        },
        num_proc=1
    )

    print('Saving Dataset... ')

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    test_dataset.save_to_disk(test_path)