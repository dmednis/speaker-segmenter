import numpy as np
from keras.utils import Sequence


class SpeechSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, timeseries_length):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.timeseries_length = timeseries_length

    def __len__(self):
        return int(np.floor(len(self.x) / float(self.timeseries_length) / float(self.batch_size)))

    def __getitem__(self, idx):
        indices = np.arange(idx * self.timeseries_length * self.batch_size,
                            idx * self.timeseries_length * self.batch_size + (self.timeseries_length * self.batch_size),
                            self.timeseries_length)
        batch_x = np.ndarray((self.batch_size, self.timeseries_length) + self.x.shape[:])
        batch_y = np.ndarray((self.batch_size, ) + self.y.shape[1:])
        for i, t in enumerate(indices):
            batch_x[i] = self.x[t: t + self.timeseries_length]
            batch_y[i] = self.y[t: t + self.timeseries_length]

        batch_x = np.reshape(batch_x, (self.batch_size, self.timeseries_length, 1) + self.x.shape[1:])

        return batch_x, batch_y
