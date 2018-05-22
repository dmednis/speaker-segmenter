import os
import librosa
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

from utils import extract_features, extract_features_v2, flatten

audio_filename = "./samples/saeima.wav"
# features_filename = "./samples/saeima-features.npy"
# predictions_filename = "./samples/saeima-predictions.npy"

data, sr = librosa.load(audio_filename)
print("SAMPLE RATE", sr)
print("DATA SHAPE", data.shape)
features = extract_features(data, sr)
features2 = extract_features_v2(data, sr)
# np.save(features_filename, features)

print("FEATURES SHAPE", features.shape)
print("FEATURES 2 SHAPE", features2.shape)

exit(0)

model = load_model('models/2018-05-22_13-29/model_vad2.15.hdf5')

series = 44

voice_fragment_count = int(np.floor(len(features) / series))
valid_voice_length = voice_fragment_count * series
features = features[:valid_voice_length]
features = np.array(np.split(features, voice_fragment_count))

predictions = model.predict(features, verbose=1)

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
