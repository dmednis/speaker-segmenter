from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np

from utils import extract_features_melspec, degrade, flatten, preload_degradations


def main():
    audio_filename = "./samples/speech-test.wav"

    data, sr = librosa.load(audio_filename)

    print("DATA", data.shape, data.shape[0] / sr)

    features = extract_features_melspec(data, sr)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(features, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    preload_degradations()
    main()
