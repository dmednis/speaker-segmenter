import shutil
import zipfile
import librosa
import glob
import os
import numpy as np

from utils import load_and_concat, extract_features, degrade, ensure_dirs


basepath = "./speech_raw/timit/data/lisa/data/timit/raw/TIMIT/"
datasets = ["TEST", "TRAIN"]


def write_to_disk(features, speaker_count):
    np.save("./speakers/TIMIT/{idx}"
            .format(idx=speaker_count),
            features)


def prepare_timit():
    speaker_count = 0
    for dataset in datasets:
        dataset_dir = basepath + dataset + "/"
        for group in os.listdir(dataset_dir):
            group_dir = dataset_dir + group + "/"
            if os.path.isdir(group_dir):
                for speaker in os.listdir(group_dir):
                    speaker_dir = group_dir + speaker + "/"
                    if os.path.isdir(speaker_dir):
                        concated, sr = load_and_concat(speaker_dir)
                        with_degraded = degrade(concated, sr)
                        features = extract_features(with_degraded, sr)
                        write_to_disk(features, speaker_count)
                        print("Extracted features - TIMIT - speaker: %i" % speaker_count)
                        speaker_count += 1


def unzip():
    zip_ref = zipfile.ZipFile("./TIMIT.zip", 'r')
    zip_ref.extractall("./timit")
    zip_ref.close()


def clean():
    shutil.rmtree("./timit", ignore_errors=True)
    shutil.rmtree("./speakers", ignore_errors=True)


def setup():
    ensure_dirs(["./speakers", "./speakers/TIMIT"])


def main():
    # clean()
    # unzip()
    setup()
    prepare_timit()


if __name__ == '__main__':
    main()
