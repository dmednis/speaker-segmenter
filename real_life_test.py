import os
import librosa
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime

from utils import extract_features_melspec, extract_features_mfcc, flatten

name = "speech"
run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# audio_filename = "noise_raw/degradations/applause.wav"
audio_filename = "./samples/speech-test.wav"

data, sr = librosa.load(audio_filename)
print("SAMPLE RATE", sr)
print("DATA SHAPE", data.shape)
features = extract_features_melspec(data, sr)
np.save("./samples/" + name + "_" + run, features)
print("FEATURES SHAPE", features.shape)

model = load_model('models/vad2_2018-05-25_13-32/model_vad2.33.hdf5')

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

np.save("./samples/predictions_" + name + "_" + run, predictions)

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
