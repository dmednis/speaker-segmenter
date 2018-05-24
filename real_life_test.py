import os
import librosa
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

from utils import extract_features_melspec, extract_features_mfcc, flatten

# audio_filename = "./samples/saeima.wav"
audio_filename = "./samples/speech-test.wav"
# features_filename = "./samples/saeima-features.npy"
# predictions_filename = "./samples/saeima-predictions.npy"

data, sr = librosa.load(audio_filename)
print("SAMPLE RATE", sr)
print("DATA SHAPE", data.shape)
features = extract_features_melspec(data, sr)

print("FEATURES SHAPE", features.shape)

model = load_model('models/vad2_2018-05-23_12-26/model_vad2.06.hdf5')

timeseries_length = 100
hop_length = 25

length = 0
remainder = len(features)
while remainder >= timeseries_length:
    length += 1
    remainder -= hop_length

x = np.ndarray((length, timeseries_length, features.shape[1]))

for i in range(length):
    x[i] = features[i * hop_length:i * hop_length + timeseries_length]

predictions = model.predict(x, verbose=1)

print("PREDICTIONS SHAPE", predictions.shape)

# np.save(predictions_filename, predictions)

predictions = flatten(flatten(predictions))

print("PREDICTIONS SHAPE", predictions.shape)
print("PREDICTIONS", predictions)

for i, pred in enumerate(predictions):
    if pred >= 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0


plt.figure(1)
plt.subplot(211)
plt.plot(data)

plt.subplot(212)
plt.plot(predictions)
plt.show()
