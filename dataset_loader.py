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


def generate_vad_voice(folder):
    speakers = glob.glob("./speakers/" + folder + "/*.npy")
    all_speakers = []
    for speaker in speakers:
        data = np.load(speaker)
        all_speakers.append(data)
    return flatten(all_speakers)


def generate_vad_voice_train():
    return flatten([generate_vad_voice("TIMIT"), generate_vad_voice("LJ")])


def generate_vad_voice_test():
    return generate_vad_voice("LIBRISPEECH")


def vad_voice_train():
    return load_or_generate("./speakers/vad_voice_train.npy", generate_vad_voice_train)


def vad_voice_test():
    return load_or_generate("./speakers/vad_voice_test.npy", generate_vad_voice_test)


def vad_noise_train():
    urbansounds = np.load("./noise/vad_noise.npy")
    return urbansounds


def vad_noise_test():
    ambience = np.load("./noise/ambience.npy")
    return ambience


if __name__ == "__main__":
    print("MAIN")
