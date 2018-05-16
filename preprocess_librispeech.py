import shutil
import os
import numpy as np

from utils import load_and_concat, extract_features, degrade, ensure_dirs, rm_dirs


def write_to_disk(features, speaker_count):
    np.save("./speakers/LIBRISPEECH/{idx}"
            .format(idx=speaker_count),
            features)


def prepare_librispeech():
    libredir = "./voice_raw/librispeech/dev-clean/"
    speaker_count = 0
    for speaker in os.listdir(libredir):
        speaker_dir = libredir + speaker + "/"
        if os.path.isdir(speaker_dir):
            concated, sr = load_and_concat(speaker_dir + "/**/*.flac")
            features = extract_features(concated, sr)
            write_to_disk(features, speaker_count)
            print("Extracted features - LIBRE - speaker: %i" % speaker_count)
            speaker_count += 1


def clean():
    rm_dirs(["./speakers/LIBRISPEECH"])


def setup():
    ensure_dirs(["./speakers", "./speakers/LIBRISPEECH"])


def main():
    setup()
    prepare_librispeech()


if __name__ == '__main__':
    main()
