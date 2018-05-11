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
        return int(np.floor(min(len(self.voice), len(self.noise)) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx % 2 == 0:
            batch_x = self.noise[idx * self.batch_size:idx * self.batch_size + self.batch_size]
            batch_y = []
            for i in range(len(batch_x)):
                batch_y[i] = [0, 1]
            batch_y = np.array(batch_y)
        else:
            batch_x = self.voice[(idx-1) * self.batch_size:(idx-1) * self.batch_size + self.batch_size]
            batch_y = []
            for i in range(len(batch_x)):
                batch_y[i] = [1, 0]
            batch_y = np.array(batch_y)

        return batch_x, batch_y
