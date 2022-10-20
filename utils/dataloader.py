import os
from typing import Any

import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset


class DataGeneratorPreLoaded(Dataset):
    def __init__(
        self,
        dataset,
        embedding_column: str,
        label_column: str
    ):
        self.data = dataset
        self.embedding_column = embedding_column
        self.label_column = label_column

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        return self.data[self.embedding_column][index], self.data[self.label_column][index]


class DataGenerator(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_sampling_rate: int,
        base_wav_path: str,
        processor: Any,
        whiper_encoder: Any,
        filename_column: str,
        label_column: str
    ):
        self.data = df
        self.base_wav_path = base_wav_path
        self.target_sampling_rate = target_sampling_rate

        self.processor = processor
        self.whiper_encoder = whiper_encoder

        self.filename_column = filename_column
        self.label_column = label_column


    def __len__(self):
        return len(self.data)


    def _load_wav(self, filename: str):
        waveform, source_sr = torchaudio.load(os.path.join(self.base_wav_path, filename))

        # Convert to mono channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Convert source sampling rate to a target sampling rate
        if source_sr != self.target_sampling_rate:
            transform = torchaudio.transforms.Resample(source_sr, self.target_sampling_rate)
            waveform = transform(waveform)

        waveform = waveform.squeeze(0)

        return waveform, self.target_sampling_rate


    def __getitem__(self, index):
        datum = self.data.iloc[index]
        waveform, sr = self._load_wav(filename=datum[self.filename_column])

        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.whiper_encoder(
                **inputs,
                output_hidden_states=False,
                output_attentions=False
            )

        z = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

        return z, datum[self.label_column]