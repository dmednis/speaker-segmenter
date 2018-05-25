import numpy as np
from keras.utils import Sequence

from utils import flatten
from dataset_loader import seg_speakers_train


class SEGSequence(Sequence):

    def __init__(self, speakers, batch_size, timeseries_length=100, hop_length=25, name=""):
        print("Initializing SEG sequence " + name)
        self.speakers = speakers
        self.batch_size = batch_size
        self.timeseries_length = timeseries_length
        self.hop_length = hop_length

        total = 0
        labels = []
        for speaker in speakers:
            total += speaker.shape[0]
            sp_labels = []
            for i in range(len(speaker)):
                if i < 10 or i >= len(speaker) - 10:
                    sp_labels.append([1])
                else:
                    sp_labels.append([0])
            labels.append(np.array(sp_labels))

        self.x = flatten(np.array(speakers))
        self.y = flatten(np.array(labels))

        print("Sequence " + name + " cleanup start")

        self.speakers = None
        del self.speakers

        print("Sequence " + name + " cleanup done")

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


if __name__ == "__main__":
    speakers = seg_speakers_train()
    gen = SEGSequence(speakers, 100)
    print("MAIN")