import shutil
import zipfile
import librosa
import glob
import os
import numpy as np

from utils import extract_features_melspec, ensure_dirs, flatten, load_and_concat


def prepare_file():
    filepath = "./noise_raw/ambient-silence.wav"
    print("Loading: " + filepath)
    data, sr = librosa.load(filepath)
    print("Extracting features: " + filepath)
    features = extract_features_melspec(data, sr)
    print(features.shape)
    np.save("./noise/ambient-silence", features)


def prepare_file2():
    filepath = "./noise_raw/degradations/*.wav"
    print("Loading: ambient sounds")
    data, sr = load_and_concat(filepath)
    print("Extracting features: " + filepath)
    features = extract_features_melspec(data, sr)
    print(features.shape)
    np.save("./noise/ambient-sounds", features)


def clean():
    shutil.rmtree("./noise", ignore_errors=True)


def setup():
    ensure_dirs(["./noise"])


def main():
    # clean()
    setup()
    prepare_file()
    prepare_file2()


if __name__ == '__main__':
    main()
