import shutil
import zipfile
import librosa
import glob
import os
import numpy as np

from utils import extract_features_melspec, ensure_dirs, flatten

filepath = "./noise_raw/ambient-silence.wav"


def prepare_file():
    print("Loading: " + filepath)
    data, sr = librosa.load(filepath)
    print("Extracting features: " + filepath)
    features = extract_features_melspec(data, sr)
    print(features.shape)
    np.save("./noise/ambience", features)


def clean():
    shutil.rmtree("./noise", ignore_errors=True)


def setup():
    ensure_dirs(["./noise"])


def main():
    # clean()
    setup()
    prepare_file()


if __name__ == '__main__':
    main()
