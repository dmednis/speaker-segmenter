import shutil
import zipfile
import librosa
import glob
import os
import numpy as np

from audio_degrader import mix_with_sound_data

basepath = "./timit/data/lisa/data/timit/raw/TIMIT/"
datasets = ["TEST", "TRAIN"]
degradation_samples = [("ambience-pub.wav", 11), ("applause.wav", 12),
                       ("brown-noise.wav", 0.01), ("white-noise.wav", 10)]
preloaded_degradations = []


def preload_degradations():
    for (degradation, value) in degradation_samples:
        z, _ = librosa.core.load("./degradations/" + degradation, mono=True)
        preloaded_degradations.append((z, value))


def flatten(fragments):
    global flat
    flat = []
    for fragment in fragments:
        flat = np.concatenate((flat, fragment))
    return flat


def permutate(fragments):
    permutations = []
    for perm in range(1):
        permutation = flatten(np.random.permutation(fragments))
        permutations.append(permutation)
    return permutations


def degrade(data, sr):
    degraded = [data]
    for (degradation, value) in preloaded_degradations:
        degradation = mix_with_sound_data(data, degradation, value)
        degraded.append(degradation)
    return degraded


def write_to_disk(features, speaker_count, dataset):
    np.save("./speakers/{dataset}/{count}"
            .format(dataset=dataset.lower(), count=speaker_count),
            features)


def load_and_concat(speaker_dir):
    fragments = []
    sr = 0
    for audio in glob.iglob(speaker_dir + "*.WAV"):
        data, _sr = librosa.load(audio)
        sr = _sr
        fragments.append(data)
    return flatten(fragments), sr


def extract_features(data, sr):
    features = []
    for d in data:
        fragment_size = sr * 3
        fragment_count = np.int32(np.floor(len(d) / fragment_size))
        timeseries_length = 128
        x = np.zeros((fragment_count, timeseries_length, 33), dtype=np.float64)
        for i in range(fragment_count):
            fragment = d[i*fragment_size:i*fragment_size+fragment_size]
            mfcc = librosa.feature.mfcc(y=fragment, sr=sr, n_mfcc=13, hop_length=512)
            spectral_center = librosa.feature.spectral_centroid(y=fragment, sr=sr, hop_length=512)
            chroma = librosa.feature.chroma_stft(y=fragment, sr=sr, hop_length=512)
            spectral_contrast = librosa.feature.spectral_contrast(y=fragment, sr=sr, hop_length=512)
            x[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            x[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            x[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            x[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
        features.append(x)
    return features


def prepare_dataset():
    for dataset in datasets:
        dataset_dir = basepath + dataset + "/"
        speaker_count = 0
        for group in os.listdir(dataset_dir):
            group_dir = dataset_dir + group + "/"
            if os.path.isdir(group_dir):
                for speaker in os.listdir(group_dir):
                    speaker_dir = group_dir + speaker + "/"
                    if os.path.isdir(speaker_dir):
                        concated, sr = load_and_concat(speaker_dir)
                        with_degraded = degrade(concated, sr)
                        features = extract_features(with_degraded, sr)
                        write_to_disk(features, speaker_count, dataset)
                        print("Extracted features - dataset: %s - speaker: %i" % (dataset, speaker_count))
                        speaker_count += 1


def unzip():
    zip_ref = zipfile.ZipFile("./TIMIT.zip", 'r')
    zip_ref.extractall("./timit")
    zip_ref.close()


def clean():
    shutil.rmtree("./timit", ignore_errors=True)
    shutil.rmtree("./speakers", ignore_errors=True)


def setup():
    if not os.path.exists("./speakers/train"):
        os.makedirs("./speakers/train")
    if not os.path.exists("./speakers/test"):
        os.makedirs("./speakers/test")


def main():
    # clean()
    # unzip()
    setup()
    preload_degradations()
    prepare_dataset()


if __name__ == '__main__':
    main()
