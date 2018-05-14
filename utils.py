import os
import shutil
import librosa
import numpy as np
import glob

from audio_degrader import mix_with_sound_data


def ensure_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def rm_dirs(dirs):
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)


def flatten(fragments):
    total = 0
    for sound in fragments:
        total += sound.shape[0]
    flat = np.ndarray((total,) + fragments[0].shape[1:])
    pointer = 0
    for i, sound in enumerate(fragments):
        start = pointer
        end = start + sound.shape[0]
        flat[start:end] = sound
        pointer = end
    return flat


def train_test_split(data, test_split):
    samples = len(data)
    test_samples = int(np.floor(samples * test_split))
    train_samples = samples - test_samples
    return data[:train_samples], data[train_samples:]


def permutate(fragments):
    permutations = []
    for perm in range(1):
        permutation = flatten(np.random.permutation(fragments))
        permutations.append(permutation)
    return permutations


def load_and_concat(audio_dir):
    fragments = []
    sr = 0
    for audio in glob.iglob(audio_dir):
        print(audio)
        data, _sr = librosa.load(audio)
        sr = _sr
        fragments.append(data)
    return flatten(fragments), sr


def extract_features(data, sr):
    n_features = 128
    melspec = librosa.feature.melspectrogram(data, sr=sr, n_mels=n_features)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = np.transpose(melspec)
    return melspec


degradation_samples = [("ambience-pub.wav", 11), ("applause.wav", 12),
                       ("brown-noise.wav", 0.01), ("white-noise.wav", 10)]
preloaded_degradations = []


def preload_degradations():
    if len(preloaded_degradations) == 0:
        for (degradation, value) in degradation_samples:
            z, _ = librosa.core.load("./noise_raw/degradations/" + degradation, mono=True)
            preloaded_degradations.append((z, value))


def degrade(data, sr):
    preload_degradations()
    degraded = [data]
    for (degradation, value) in preloaded_degradations:
        degradation = mix_with_sound_data(data, degradation, value)
        degraded.append(degradation)
    return flatten(degraded)
