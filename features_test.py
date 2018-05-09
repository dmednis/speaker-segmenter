from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np

from preprocess_speech import load_and_concat, degrade, flatten, preload_degradations


def extract_features(data, sr):
    features = []
    for d in data:
        fragment_size = sr * 3
        fragment_count = np.int32(np.floor(len(d) / fragment_size))
        timeseries_length = 128
        x = np.zeros((fragment_count, timeseries_length, 33), dtype=np.float16)
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


def main():
    concated, sr = load_and_concat("timit/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FALR0/")

    with_degraded = degrade(concated, sr)
    data = flatten(with_degraded)

    print("DATA", data.shape, data.shape[0] / sr)

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=32, hop_length=512)
    mfcc = np.transpose(mfcc)

    print(mfcc.shape)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    # librosa.output.write_wav('features_test.wav', data, 22050)
    #
    # # And compute the spectrogram magnitude and phase
    # S_full = librosa.stft(data)
    #
    # print("STFT", np.transpose(S_full).shape)
    # #
    # rp = np.max(np.abs(S_full))
    #
    # plt.figure(figsize=(12, 8))
    #
    # plt.subplot(3, 1, 1)
    # librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=rp), y_axis='log', x_axis='time')
    # plt.colorbar()
    # plt.title('Full spectrogram')
    # plt.show()


if __name__ == '__main__':
    preload_degradations()
    main()
