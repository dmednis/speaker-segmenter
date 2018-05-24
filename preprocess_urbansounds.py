import shutil
import zipfile
import librosa
import glob
import os
import numpy as np

from utils import extract_features_melspec, ensure_dirs, flatten, shuffle


def prepare_urbansounds():
    features = []
    files = glob.glob("./noise_raw/urbansounds/data/**/*.wav")
    for i, file in enumerate(files):
        print(str(i + 1) + " loading: " + file)
        try:
            data, _sr = librosa.load(file)
        except:
            continue
        sr = _sr
        f = extract_features_melspec(data, sr)
        features.append(f)
        if i > 0 and i % 100 == 0:
            np.save("./noise/vad_noise_" + str(i), flatten(features))
    shuffle(features)
    features = flatten(features)
    print(features.shape)
    np.save("./noise/vad_noise", features)


def clean():
    shutil.rmtree("./noise", ignore_errors=True)


def setup():
    ensure_dirs(["./noise"])


def main():
    # clean()
    setup()
    prepare_urbansounds()


if __name__ == '__main__':
    main()
