from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np

from make_timit import load_and_concat, degrade, flatten, preload_degradations


def main():
    concated, sr = load_and_concat("timit/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FALR0/")

    with_degraded = degrade(concated, sr)
    data = flatten(with_degraded)

    print("DATA", data.shape, data.shape[0] / sr)

    librosa.output.write_wav('features_test.wav', data, 22050)

    # And compute the spectrogram magnitude and phase
    S_full = librosa.stft(data[:1500000])

    print("STFT", np.transpose(S_full).shape)

    rp = np.max(np.abs(S_full))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=rp), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Full spectrogram')
    plt.show()

if __name__ == '__main__':
    preload_degradations()
    main()
