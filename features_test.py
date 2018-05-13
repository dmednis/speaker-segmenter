from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np

from preprocess_timit import load_and_concat, degrade, flatten, preload_degradations


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
    # concated, sr = load_and_concat("./voice_raw/timit/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FALR0/")
    concated, sr = load_and_concat("./noise_raw/urbansounds/data/siren/16772.wav")

    print(concated)
    print(sr)

    # with_degraded = degrade(concated, sr)
    data = np.array(concated)

    print("DATA", data.shape, data.shape[0] / sr)

    S = librosa.feature.melspectrogram(data, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    print(log_S.shape)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    preload_degradations()
    main()
