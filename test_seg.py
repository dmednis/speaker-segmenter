import numpy as np
from matplotlib import pyplot as plt
import librosa

from postprocess_utils import seg_metrics
from utils import extract_features_melspec

audio_filename = "./samples/seg-test16.wav"
features_filename = "./samples/seg-test_features.npy"
# predictions_filename = "samples/predictions_2018-05-24_17-48.npy"

audio, sr = librosa.load(audio_filename, sr=16000)
# predictions = np.load(predictions_filename)
# features = np.load(features_filename)
features = extract_features_melspec(audio, sr)

print("AUDIO", audio.shape)
# print("PREDICTIONS", predictions.shape)
print("FEATURES", features.shape)

timeseries_length = 100
hop_length = 25

# preds = deoverlap_predictions(predictions, features, hop_length)
# norm_preds = defragment_vad(preds)

# reference = [(6.42, 6.85), (13.49, 13.78)]
reference = [(0, 6.42), (6.42, 13.49), (13.49, 20.43)]

# lium = [(13.55, 13.67)]
lium = [(0, 13.55), (13.55, 20.43)]

ref_plot = [0.1 for _ in range(len(audio))]
for r in reference:
    sr = 16000
    (start, end) = librosa.core.time_to_samples(r, sr=sr)
    start = max((0, start))
    end = min((len(audio), end))
    print("REF", start, end)
    ref_plot[start:end] = [0.9 for _ in range(end - start)]
print(len(ref_plot))


lium_seg = [0 for _ in range(len(audio))]
for l in lium:
    sr = 16000
    (start, end) = librosa.core.time_to_samples(l, sr=sr)
    start = max((0, start))
    end = min((len(audio), end))
    print("LIUM", start, end)
    lium_seg[start:end] = [1 for _ in range(end - start)]
print(len(lium_seg))

seg_metrics(lium, reference)

fig, (
    (ax1),
    (ax2),
    # (ax3)
) = plt.subplots(2, 1)

ax1.plot(audio)
ax1.set_title('skaņas līkne', fontsize='large')

ax2.plot(lium_seg)
ax2.plot(ref_plot)
ax2.set_title('LIUM rezultāti', fontsize='large')

# ax3.plot(norm_preds)
# ax3.plot(ref_plot)
# ax3.set_title('normalizēti rezultāti', fontsize='large')

plt.show()



