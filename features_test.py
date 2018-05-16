from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np

from utils import load_and_concat, degrade, flatten, preload_degradations


def main():
    concated, sr = load_and_concat("./voice_raw/timit/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FALR0/*.WAV")
    # concated, sr = load_and_concat("./noise_raw/urbansounds/data/siren/16772.wav")

    print(concated)
    print(sr)
    sec = len(concated) / sr
    print(sec)

    with_degraded = degrade(concated, sr)
    data = with_degraded

    print("DATA", data.shape, data.shape[0] / sr)

    S = librosa.feature.melspectrogram(data, sr=sr, n_mels=128, fmax=8000, power=1)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    print(log_S.shape)

    transposed = np.transpose(log_S)

    print(len(transposed) / sec)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    preload_degradations()
    main()
