import os
import random
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

        # print(final_embeddings.shape, "-"*50)

        if self.add_gaussian_noise:
            final_embeddings = self._add_gaussian_noise(signal=final_embeddings, shape=final_embeddings.shape)


        return final_embeddings, label


class DataGenerator(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoder: Any,
        processor: Any,
        label_column: str,
        use_pooling: bool,
        base_wav_path: str,
        filename_column: str,
        target_sampling_rate: int
    ):
        self.data = df
        self.base_wav_path = base_wav_path
        self.target_sampling_rate = target_sampling_rate

        self.processor = processor
        self.encoder = encoder

        self.filename_column = filename_column
        self.label_column = label_column

        self.use_pooling = use_pooling


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
        # print(inputs)
        print(inputs["input_features"].shape)
        with torch.no_grad():
            outputs = self.encoder(**inputs)

        if self.use_pooling:
            z = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        else:
            z = outputs.last_hidden_state

        # print(index, z.shape, datum[self.label_column])

        return z, datum[self.label_column]


class DataGeneratorForWhisper(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: Any,
        label_column: str,
        base_wav_path: str,
        filename_column: str,
        target_sampling_rate: int,
        use_mixup: bool,
        mixup_alpha: float,
        use_specaug: bool,
        specaug_freqm: int,
        specaug_timem: int,
        data_type: str,
        class_num: int,
        use_hot_one_encoding: bool
    ):
        self.data = df
        self.base_wav_path = base_wav_path
        self.target_sampling_rate = target_sampling_rate

        self.processor = processor

        self.filename_column = filename_column
        self.label_column = label_column

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.use_specaug = use_specaug
        self.specaug_freqm = specaug_freqm
        self.specaug_timem = specaug_timem

        self.class_num = class_num

        self.data_type = data_type

        self.use_hot_one_encoding = use_hot_one_encoding


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
        # print(index)
        if self.use_mixup and self.data_type=="train":
            datum = self.data.iloc[index]
            # Samples a random audio from the dataset to do mixup
            rand_index = random.randint(0, len(self.data)-1)
            rand_datum = self.data.iloc[rand_index]

            # Makes sure that the class from the random audio is different than the main audio
            while datum[self.label_column] == rand_datum[self.label_column]:
                rand_index = random.randint(0, len(self.data)-1)
                rand_datum = self.data.iloc[rand_index]
                # print("-"*5, index)

            # Cut audios
            audio_original, sr = self._load_wav(filename=datum[self.filename_column])
            audio_rand, _ = self._load_wav(filename=rand_datum[self.filename_column])

            # Cut or pad audios
            if audio_original.shape[0] != audio_rand.shape[0]:
                if audio_original.shape[0] > audio_rand.shape[0]:
                    # padding
                    temp_wav = torch.zeros(audio_original.shape[0])
                    temp_wav[:audio_rand.shape[0]] = audio_rand
                    audio_rand = temp_wav
                else:
                    # cutting
                    audio_rand = audio_rand[:audio_original.shape[0]]

            # Sample lambda from a beta distribution based on the value of alpha
            mix_lambda = torch.tensor(np.random.beta(self.mixup_alpha, self.mixup_alpha))

            # Mixup
            waveform = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # Hot one encoding for mixup
            label = np.zeros(self.class_num, dtype='f')
            label[datum[self.label_column]] = mix_lambda
            label[rand_datum[self.label_column]] = 1 - mix_lambda

        else:
            datum = self.data.iloc[index]
            waveform, sr = self._load_wav(filename=datum[self.filename_column])
            label = datum[self.label_column]

            if self.use_hot_one_encoding:
                # Hot one encoding for mixup
                temp_label = np.zeros(self.class_num, dtype='f')
                temp_label[label] = 1
                label = temp_label

            # print(label, datum[self.label_column])

        # Shape: (1, 80, 3000) -> (1, freq, time)
        fbank = self.processor(waveform, sampling_rate=sr, return_tensors="pt")["input_features"]

        # SpecAugmentation
        if self.use_specaug and self.data_type=="train":
            freqm = torchaudio.transforms.FrequencyMasking(self.specaug_freqm)
            timem = torchaudio.transforms.TimeMasking(self.specaug_timem)
            if self.specaug_freqm != 0:
                fbank = freqm(fbank)
            if self.specaug_timem != 0:
                fbank = timem(fbank)

        # Shape: (80, 3000) -> (freq, time)
        fbank = fbank.squeeze(0)

        return fbank, label