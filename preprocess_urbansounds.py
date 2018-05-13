import shutil
import zipfile
import librosa
import glob
import os
import numpy as np


def extract_features(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=32, hop_length=512)
    mfcc = np.transpose(mfcc)
    return mfcc


def prepare_urbansounds():
    features = []
    files = glob.glob("./urbansounds/data/**/*.wav")
    for i, file in enumerate(files):
        print(str(i + 1) + "loading: " + file)
        try:
            data, _sr = librosa.load(file)
        except:
            continue
        sr = _sr
        f = extract_features(data, sr)
        features.append(f)
        if i > 0 and i % 100 == 0:
            np.save("./noise/noise_x_" + str(i), np.array(features))
    features = np.array(features)
    print(features.shape)
    np.save("./noise/noise_x", features)


def flatten_urbansounds():
    us = np.load("./noise/noise_x.npy")
    print("determining total length")
    total = 0
    for sound in us:
        total += sound.shape[0]
    flat = np.ndarray((total, us[0].shape[1]))
    print("final shape ", flat.shape)
    print("populating")
    pointer = 0
    for i, sound in enumerate(us):
        print("populating ", i)
        start = pointer
        end = start + sound.shape[0]
        flat[start:end] = sound
        pointer = end
    np.save("./noise/noise_x_flat", flat)


def clean():
    shutil.rmtree("./noise", ignore_errors=True)


def setup():
    if not os.path.exists("./noise"):
        os.makedirs("./noise")


def main():
    # clean()
    # setup()
    # prepare_urbansounds()
    flatten_urbansounds()


if __name__ == '__main__':
    main()
