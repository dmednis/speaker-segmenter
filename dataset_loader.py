import os
import glob
import numpy as np

from utils import flatten


def load_or_generate(filename, generator):
    if os.path.isfile(filename):
        print("Loading " + filename)
        dataset = np.load(filename, mmap_mode="r")
        return dataset
    else:
        print("Generating " + filename)
        dataset = generator()
        # np.save(filename, dataset)
        return dataset


def generate_vad_voice(folder):
    speakers = glob.glob("./speakers/" + folder + "/*.npy")
    all_speakers = []
    for speaker in speakers:
        data = np.load(speaker)
        all_speakers.append(data)
    return flatten(all_speakers)


def generate_seg_speakers(folder):
    speakers = glob.glob("./speakers/" + folder + "/*.npy")
    all_speakers = []
    for speaker in speakers:
        data = np.load(speaker)
        all_speakers.append(data)
    return np.array(all_speakers)


def generate_vad_voice_train():
    return flatten([generate_vad_voice("TIMIT"), generate_vad_voice("LJ")])


def generate_vad_voice_test():
    return generate_vad_voice("LIBRISPEECH")


def vad_voice_train():
    return load_or_generate("./speakers/vad_voice_train.npy", generate_vad_voice_train)


def vad_voice_test():
    return load_or_generate("./speakers/vad_voice_test.npy", generate_vad_voice_test)


def vad_noise_train():
    urbansounds = np.load("./noise/vad_noise.npy", mmap_mode="r")
    return urbansounds


def vad_noise_test():
    ambient_silence = np.load("./noise/ambient-silence.npy", mmap_mode="r")
    ambient_sounds = np.load("./noise/ambient-sounds.npy", mmap_mode="r")
    return flatten([ambient_silence, ambient_sounds])


def generate_seg_speakers_train():
    return generate_seg_speakers("TIMIT")


def generate_seg_speakers_test():
    return generate_seg_speakers("LIBRISPEECH")


def seg_speakers_train():
    return load_or_generate("./speakers/seg_speakers_train.npy", generate_seg_speakers_train)


def seg_speakers_test():
    return load_or_generate("./speakers/seg_speakers_test.npy", generate_seg_speakers_test)


if __name__ == "__main__":
    seg_speakers_train()
    seg_speakers_test()
    print("MAIN")
