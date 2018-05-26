import numpy as np
from keras.utils import Sequence

from utils import flatten
from dataset_loader import seg_speakers_train


class SEGSequence(Sequence):

    def __init__(self, speakers, batch_size, timeseries_length=100, hop_length=25, fragment_len=500, name=""):
        print("Initializing SEG sequence " + name)
        self.speakers = speakers
        self.batch_size = batch_size
        self.timeseries_length = timeseries_length
        self.hop_length = hop_length

        split_speakers = []
        max_len = 0
        for speaker in self.speakers:
            speaker_fragment_count = int(np.floor(len(speaker) / fragment_len))
            valid_speaker_length = speaker_fragment_count * fragment_len
            speaker_ = speaker[:valid_speaker_length]
            fragments = np.array(np.split(speaker_, speaker_fragment_count))
            max_len = np.max((max_len, speaker_fragment_count))
            split_speakers.append(fragments)

        spks = []
        last_inserted = -1
        for i in range(max_len):
            for j in range(len(split_speakers)):
                if i < len(split_speakers[j]) and last_inserted != j:
                    spks.append(split_speakers[j][i])
                    last_inserted = j

        spks = np.array(spks)
        print(spks.shape)

        self.ones = 0
        self.zeros = 0

        total = 0
        labels = []
        for speaker in spks:
            total += speaker.shape[0]
            sp_labels = []
            for i in range(len(speaker)):
                if i < 25 or i >= len(speaker) - 25:
                    sp_labels.append([1])
                    self.ones += 1
                else:
                    sp_labels.append([0])
                    self.zeros += 1
            labels.append(np.array(sp_labels))

        self.x = flatten(np.array(spks))
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
        print("Ones: ", self.ones)
        print("Zeros: ", self.zeros)
        print("Ratio: ", self.zeros / self.ones)

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