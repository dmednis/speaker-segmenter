import shutil
import os
import numpy as np

from utils import load_and_concat, extract_features_melspec, degrade, ensure_dirs, rm_dirs


def write_to_disk(features, speaker_count):
    np.save("./speakers/LJ/{idx}"
            .format(idx=speaker_count),
            features)


def prepare_ljspeech():
    data, sr = load_and_concat("./voice_raw/ljspeech/wavs/LJ01*.wav")
    features = extract_features_melspec(data, sr)
    write_to_disk(features, 0)


def clean():
    rm_dirs(["./speakers/LJ"])


def setup():
    ensure_dirs(["./speakers", "./speakers/LJ"])


def main():
    setup()
    prepare_ljspeech()


if __name__ == '__main__':
    main()
