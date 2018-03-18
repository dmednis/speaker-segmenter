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


def degrade(audio_list, sr):
    degradations = []
    for audio in audio_list:
        for (degradation, value) in preloaded_degradations:
            degraded = mix_with_sound_data(audio, degradation, value)
            degradations.append(degraded)
    return degradations


def write_to_disk(permutations, degradations, speaker_count, dataset):
    for idx, permutation in enumerate(permutations):
        np.save("./speakers/{dataset}/sp_{count}_{permutation}"
                .format(dataset=dataset.lower(), count=speaker_count, permutation=idx),
                permutation)
    for idx, degradation in enumerate(degradations):
        np.save("./speakers/{dataset}/sp_{count}_{permutation}_deg_{degradation}"
                .format(dataset=dataset.lower(), count=speaker_count, permutation=idx // 4, degradation=idx % 4),
                degradation)


def load_and_concat(speaker_dir):
    fragments = []
    sr = 0
    for audio in glob.iglob(speaker_dir + "*.WAV"):
        data, _sr = librosa.load(audio)
        sr = _sr
        fragments.append(data)
    return fragments, sr


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
                        print(dataset, speaker_count)
                        fragments, sr = load_and_concat(speaker_dir)
                        permutations = permutate(fragments)
                        degradations = degrade(permutations, sr)
                        write_to_disk(permutations, degradations, speaker_count=speaker_count, dataset=dataset)
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
