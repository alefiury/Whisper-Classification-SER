import os
from typing import Any, Optional, Tuple, List

import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset


class DataGeneratorPreLoaded(Dataset):
    def __init__(
        self,
        label_column: str,
        embedding_column: str,
        dataset: Any = None,
        dataset2: Optional[Any] = None,
        file_path_column: Optional[str] = None,
        concatanation_type: Optional[str] = None, # possible values: 'time', 'channel',
        add_gaussian_noise: bool = False
    ):
        self.data = dataset
        self.embedding_column = embedding_column
        self.label_column = label_column

        self.data2 = dataset2
        self.file_path_column = file_path_column
        self.concatanation_type = concatanation_type

        self.add_gaussian_noise = add_gaussian_noise

    def __len__(self):
        return self.data.num_rows


    def _add_gaussian_noise(
        self,
        signal: List[float],
        min_amplitude: float = 0.001,
        max_amplitude: float = 0.015,
        shape: Tuple = (1, 512)
    ) -> List[float]:
        amplitude = torch.tensor(np.random.uniform(low=min_amplitude, high=max_amplitude))
        noise = torch.randn(shape)

        augmented_signal = signal + noise*amplitude

        return augmented_signal


    def __getitem__(self, index):
        label = self.data[self.label_column][index]

        if self.data2 is not None:
            assert self.data[self.file_path_column][index] == self.data2[self.file_path_column][index]

            if self.concatanation_type == "time":
                embeddings1 = self.data[self.embedding_column][index]
                embeddings2 = self.data2[self.embedding_column][index]

                final_embeddings = torch.cat((embeddings1, embeddings2), 0).unsqueeze(0)

            elif self.concatanation_type == "channel":
                embeddings1 = F.pad(self.data[self.embedding_column][index], (0, 512)).unsqueeze(0)
                embeddings2 = self.data2[self.embedding_column][index].unsqueeze(0)

                final_embeddings = torch.cat((embeddings1, embeddings2), 0)

            else:
                print("Concatenation type is not supported; embeddings from self.data will be returned instead")

        else:
            final_embeddings = self.data[self.embedding_column][index].unsqueeze(0)

        if self.add_gaussian_noise:
            final_embeddings = self._add_gaussian_noise(signal=final_embeddings, shape=final_embeddings.shape)


        return final_embeddings, label


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