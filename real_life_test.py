import os
import librosa
import numpy as np
from keras.models import load_model

from make_timit import extract_features

audio_filename = "saeima.mp3"
features_filename = "saeima-features.npy"
predictions_filename = "saeima-predictions.npy"

if os.path.isfile(features_filename):
    features = np.load(features_filename)
else:
    data, sr = librosa.load(audio_filename)
    print("SAMPLE RATE", sr)
    print("DATA SHAPE", data.shape)
    features = extract_features(data, sr)
    np.save(features_filename, features)

print("FEATURES SHAPE", features.shape)

model = load_model('model-experiment.hdf5')

predictions = model.predict(features, verbose=1)

print("PREDICTIONS SHAPE", predictions.shape)

np.save(predictions_filename, predictions)

speaker_changes = []
for idx, frame in enumerate(predictions):
    # print("==================FRAME ", idx, " AT ", (idx * 3) // 60, ":", (idx * 3) % 60)
    change = round(frame[0], 2)
    not_change = round(frame[1], 2)
    if change > not_change:
        speaker_changes.append(idx)
        print(change, not_change)
        print("SPEAKER CHANGE ", idx, " AT ", (idx * 3) // 60, ":", (idx * 3) % 60)

print("TOTAL CHANGES", len(speaker_changes))
