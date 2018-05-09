import os
import librosa
import numpy as np
from keras.models import load_model

from preprocess_speech import extract_features

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

model = load_model('models/model.03.hdf5')

predictions = model.predict(features, verbose=1)

print("PREDICTIONS SHAPE", predictions.shape)

np.save(predictions_filename, predictions)

speaker_changes = []
for idx, frame in enumerate(predictions):
    # print("==================FRAME ", idx, " AT ", (idx * 3) // 60, ":", (idx * 3) % 60)
    no_change = round(frame[0], 2)
    voice_ends = round(frame[1], 2)
    new_voice = round(frame[2], 2)
    same_voice = round(frame[3], 2)
    if no_change < 0.5:
        speaker_changes.append(idx)
        print(no_change, new_voice, new_voice, same_voice)
        print("SPEAKER CHANGE ", idx, " AT ", (idx * 3) // 60, ":", (idx * 3) % 60)

print("TOTAL CHANGES", len(speaker_changes))
