import numpy as np
from matplotlib import pyplot as plt
import librosa

from postprocess_utils import deoverlap_predictions, defragment_vad, vad_metrics
from utils import extract_features_melspec

audio_filename = "./samples/speech-test.wav"
features_filename = "./samples/speech-test_features.npy"
predictions_filename = "samples/predictions_2018-05-24_17-48.npy"

audio, sr = librosa.load(audio_filename)
predictions = np.load(predictions_filename)
features = np.load(features_filename)
# features = extract_features_melspec(audio, sr)

print("AUDIO", audio.shape)
print("PREDICTIONS", predictions.shape)
print("FEATURES", features.shape)

timeseries_length = 100
hop_length = 25

preds = deoverlap_predictions(predictions, features, hop_length)
norm_preds = defragment_vad(preds)

reference = [(0.543876, 7.962684), (8.583042, 23.420658)]

lium = [[0.0, 3.11], [3.34, 13.15],
        [13.15, 23.8]]

ref_plot = [0.1 for _ in range(len(preds))]
for r in reference:
    sr = 22050
    (start, end) = librosa.core.time_to_frames(r, sr=sr, hop_length=(0.016 * sr), n_fft=2048)
    start = max((0, start))
    end = min((len(preds), end))
    ref_plot[start:end] = [0.9 for _ in range(end - start)]
print(len(ref_plot))


# lium_vad = [0 for _ in range(len(preds))]
# for l in lium:
#     sr = 22050
#     (start, end) = librosa.core.time_to_frames(l, sr=sr, hop_length=(0.016 * sr), n_fft=2048)
#     start = max((0, start))
#     end = min((len(preds), end))
#     lium_vad[start:end] = [1 for _ in range(end - start)]
# print(len(lium_vad))

vad_metrics(norm_preds, reference)


fig, (
    (ax1),
    (ax2),
    (ax3)
) = plt.subplots(3, 1)

ax1.plot(audio)
ax1.set_title('skaņas līkne', fontsize='large')

ax2.plot(preds)
ax2.plot(ref_plot)
ax2.set_title('LIUM rezultāti', fontsize='large')

ax3.plot(norm_preds)
ax3.plot(ref_plot)
ax3.set_title('normalizēti rezultāti', fontsize='large')

plt.show()


# DIY VAD {
# 'precision': 0.9725755158545583,
# 'error': 0.039384154023979864,
# 'recall': 0.9884890568745904,
# 'accuracy': 0.9616838403740963}

# LIUM VAD {
# 'precision': 0.9389033495670994,
# 'error': 0.07446060488651701,
# 'recall': 0.9899583121038643,
# 'accuracy': 0.9300457707975986}
