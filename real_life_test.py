import os
import librosa
import numpy as np
from keras.models import load_model

from preprocess_urbansounds import extract_features
from VADSequence import VADSequence

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

model = load_model('models/model_vad.01.hdf5')

series = 44

voice_fragment_count = int(np.floor(len(features) / series))
valid_voice_length = voice_fragment_count * series
features = features[:valid_voice_length]
features = np.array(np.split(features, voice_fragment_count))

predictions = model.predict(features, verbose=1)

print("PREDICTIONS SHAPE", predictions.shape)

np.save(predictions_filename, predictions)


for idx, frame in enumerate(predictions):
    print("==================FRAME ", idx, " AT ", (idx * 1) // 60, ":", (idx * 1) % 60)
    val = round(frame[0], 2)
    print(frame[0])
    print(val)
    if val > 0.5:
        print("*************VOICE*************")
