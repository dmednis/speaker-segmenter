import numpy as np
from keras.utils import Sequence

from utils import flatten


class VADSequence(Sequence):

    def __init__(self, voice, noise, batch_size, timeseries_length=100, hop_length=25, mix_per_amount=1000, name=""):
        print("Initializing VAD sequence " + name)
        self.voice, self.noise = voice, noise
        self.batch_size = batch_size
        self.timeseries_length = timeseries_length
        self.hop_length = hop_length

        voice_fragment_count = int(np.floor(len(self.voice) / mix_per_amount))
        valid_voice_length = voice_fragment_count * mix_per_amount
        self.voice = self.voice[:valid_voice_length]
        self.voice = np.array(np.split(self.voice, voice_fragment_count))

        noise_fragment_count = int(np.floor(len(self.noise) / mix_per_amount))
        valid_noise_length = noise_fragment_count * mix_per_amount
        self.noise = self.noise[:valid_noise_length]
        self.noise = np.array(np.split(self.noise, noise_fragment_count))

        self.x = np.ndarray((voice_fragment_count + noise_fragment_count, mix_per_amount, voice.shape[1]))
        self.y = np.ndarray((voice_fragment_count + noise_fragment_count, mix_per_amount, 1))

        pointer = 0

        for i in range(max(len(self.voice), len(self.noise))):
            voice_fragment = self.voice[i]
            self.x[pointer] = voice_fragment
            self.y[pointer] = np.array([[1] for _ in range(len(voice_fragment))])
            pointer += 1
            if i < len(self.noise):
                noise_fragment = self.noise[i]
                self.x[pointer] = noise_fragment
                self.y[pointer] = np.array([[0] for _ in range(len(noise_fragment))])
                pointer += 1

        print("Sequence " + name + " cleanup start")

        self.voice = None
        self.noise = None

        print("Sequence " + name + " cleanup done")

        self.x = flatten(self.x)
        self.y = flatten(self.y)

        self.length = 0
        remainder = len(self.x)
        while remainder >= timeseries_length:
            self.length += 1
            remainder -= hop_length

        print("Sequence " + name + " initialization done")

    def __len__(self):
        return int(np.floor(self.length / self.batch_size))

    def __getitem__(self, idx):
        batch_x = np.ndarray((self.batch_size, self.timeseries_length, self.x.shape[1]))
        batch_y = np.ndarray((self.batch_size, self.timeseries_length, self.y.shape[1]))

        batch_start = idx * self.batch_size * self.hop_length
        for i in range(self.batch_size):
            batch_x[i] = self.x[batch_start + (i * self.hop_length):batch_start + (
                        i * self.hop_length) + self.timeseries_length]
            batch_y[i] = self.y[batch_start + (i * self.hop_length):batch_start + (
                        i * self.hop_length) + self.timeseries_length]

        return batch_x, batch_y
