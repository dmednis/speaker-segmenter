import os
import glob
import numpy as np

from utils import flatten


def load_or_generate(filename, generator):
    if os.path.isfile(filename):
        print("Loading " + filename)
        dataset = np.load(filename)
        return dataset
    else:
        print("Generating " + filename)
        dataset = generator()
        np.save(filename, dataset)
        return dataset


def generate_vad_voice():
    speakers = glob.glob("./speakers/TIMIT/*.npy")
    all_speakers = []
    for speaker in speakers:
        data = np.load(speaker)
        all_speakers.append(data)
    return flatten(all_speakers)


def vad_voice():
    return load_or_generate("./speakers/vad_voice.npy", generate_vad_voice)


def vad_noise():
    return np.load("./noise/vad_noise.npy")


if __name__ == "__main__":
    vad_voice()
