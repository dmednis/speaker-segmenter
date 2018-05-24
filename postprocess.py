import numpy as np
from matplotlib import pyplot as plt
import librosa

from utils import deoverlap_predictions, defragment_vad


audio_filename = "./samples/speech-test.wav"
features_filename = "./samples/speech-test_features.npy"
predictions_filename = "samples/predictions_2018-05-24_17-48.npy"

audio, sr = librosa.load(audio_filename)
predictions = np.load(predictions_filename)
features = np.load(features_filename)

timeseries_length = 100
hop_length = 25

preds = deoverlap_predictions(predictions, features, hop_length)
preds = defragment_vad(preds)

plt.figure(1)
plt.subplot(211)
plt.plot(audio)

plt.subplot(212)
plt.plot(preds)
plt.show()