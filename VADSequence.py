import random
import numpy as np
from keras.utils import Sequence


class VADSequence(Sequence):

    def __init__(self, voice, noise, batch_size, timeseries_length):
        self.voice, self.noise = voice, noise
        self.batch_size = batch_size
        self.timeseries_length = timeseries_length

        voice_fragment_count = int(np.floor(len(self.voice) / self.timeseries_length))
        valid_voice_length = voice_fragment_count * self.timeseries_length
        self.voice = self.voice[:valid_voice_length]
        self.voice = np.array(np.split(self.voice, voice_fragment_count))

        noise_fragment_count = int(np.floor(len(self.noise) / self.timeseries_length))
        valid_noise_length = noise_fragment_count * self.timeseries_length
        self.noise = self.noise[:valid_noise_length]
        self.noise = np.array(np.split(self.noise, noise_fragment_count))

    def __len__(self):

        return int(
            min(np.floor(len(self.voice) / self.batch_size / 2), np.floor(len(self.noise) / self.batch_size / 2)))

    def __getitem__(self, idx):
        # noise_len = random.randint(0, self.batch_size)
        # voice_len = self.batch_size - noise_len

        noise_len = voice_len = int(self.batch_size / 2)

        batch_x = np.ndarray((self.batch_size, self.timeseries_length, self.voice.shape[2]))

        batch_x[:noise_len] = self.noise[idx * noise_len:idx * noise_len + noise_len]
        batch_x[noise_len:] = self.voice[idx * voice_len:idx * voice_len + voice_len]

        batch_y = []
        for i in range(len(batch_x)):
            if i < noise_len:
                batch_y.append(0)
            else:
                batch_y.append(1)
        batch_y = np.array(batch_y)

        return batch_x, batch_y
