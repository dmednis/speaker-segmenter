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


def extract_features_mfcc(data, sr):
    frame_length = int(np.floor(0.032 * sr))
    hop_length = int(np.floor(0.016 * sr))

    stft = np.abs(librosa.stft(data, win_length=frame_length, hop_length=hop_length)) ** 2
    spectrogram = librosa.feature.melspectrogram(S=stft, y=data)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=11)
    rmse = librosa.feature.rmse(S=spectrogram, frame_length=frame_length, hop_length=hop_length)

    mfcc_1 = librosa.feature.delta(mfcc)
    mfcc_2 = librosa.feature.delta(mfcc, order=2)
    rmse_1 = librosa.feature.delta(rmse)
    rmse_2 = librosa.feature.delta(rmse, order=2)

    mfcc = np.transpose(mfcc)
    mfcc_1 = np.transpose(mfcc_1)
    mfcc_2 = np.transpose(mfcc_2)
    rmse_1 = np.transpose(rmse_1)
    rmse_2 = np.transpose(rmse_2)

    frames = min(len(mfcc),
                 len(mfcc_1),
                 len(mfcc_2),
                 len(rmse_1),
                 len(rmse_2))

    features = np.zeros((len(mfcc), 35))

    for i in range(frames):
        features[i, 0:11] = mfcc[i]
        features[i, 11:22] = mfcc_1[i]
        features[i, 22:33] = mfcc_2[i]
        features[i, 33] = rmse_1[i]
        features[i, 34] = rmse_2[i]

    return features


def extract_features_melspec(data, sr):
    n_features = 64
    frame_length = int(np.floor(0.032 * sr))
    hop_length = int(np.floor(0.016 * sr))

    melspec = librosa.feature.melspectrogram(y=data, n_mels=n_features, hop_length=hop_length, n_fft=frame_length)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = np.transpose(melspec)
    melspec_d = librosa.feature.delta(melspec)
    features = np.hstack((melspec, melspec_d))
    return features


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


def shuffle(x):
    np.random.shuffle(x)


def deoverlap_predictions(predictions, features, hop_length):
    deoverlapped = [[] for i in range(len(features))]

    print(predictions.shape)
    print(features.shape)

    for i, f in enumerate(predictions):
        for j, p in enumerate(f):
            idx = (i * hop_length) + j
            if p >= 0.5:
                deoverlapped[idx].append(1)
            else:
                deoverlapped[idx].append(0)

    averaged = np.zeros(len(features))

    for i, p in enumerate(np.array(deoverlapped)):
        if len(p):
            averaged[i] = np.max(p)
        else:
            averaged[i] = 0


def defragment_vad(predictions):
    defragmented = np.zeros(len(predictions))
    mode = 0
    ones = 0
    zeros = 0
    for i, p in enumerate(predictions):
        defragmented[i] = mode
        if p == 0:
            ones = 0
            zeros += 1
            if zeros >= 50:
                mode = 0
                defragmented[i - zeros:i] = [0 for i in range(zeros)]
        else:
            ones += 1
            zeros = 1
            if ones >= 20:
                mode = 1
                defragmented[i - ones:i] = [1 for i in range(ones)]

    return defragmented
