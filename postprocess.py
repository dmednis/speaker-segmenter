import numpy as np
from matplotlib import pyplot as plt
import librosa

from postprocess_utils import deoverlap_predictions, defragment_vad, vad_metrics

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

reference = [(0.543876, 7.962684), (8.583042, 23.420658)]
lium = [(0.0,8.039999961853027), (10.640000343322754,13.15000033378601),
        (8.039999961853027,10.639999866485596), (13.149999618530273,23.809999465942383),
        (23.809999465942383,32.9399995803833), (32.939998626708984,37.64999866485596),
        (37.650001525878906,40.20000147819519)]

vad_metrics(lium, reference)

plt.figure(1)
plt.subplot(211)
plt.plot(audio)

plt.subplot(212)
plt.plot(preds)
plt.show()

# m = re.match(r"^.*#t=([\d\.,]*)>.*$", "Isaac Newton, physicist")
