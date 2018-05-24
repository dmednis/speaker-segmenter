import numpy as np


def deoverlap_predictions(predictions, features, timeseries_length, hop_length):
    deoverlapped = np.ndarray((len(features), 0))
    print(deoverlapped.shape)
    for i, frag in enumerate(predictions):
        print(frag)
        for j, pred in enumerate(frag):
            idx = (i * hop_length) + j
            features = deoverlapped[idx]
            print(idx)
            print(features)
            print(pred)
            print(np.append(features, pred))
            deoverlapped[idx] = np.append(features, pred)
        print(deoverlapped)
        exit(0)


features_filename = "./samples/speech-test_features.npy"
predictions_filename = "samples/predictions_2018-05-24_17-48.npy"

predictions = np.load(predictions_filename)
features = np.load(features_filename)

timeseries_length = 100
hop_length = 25

deoverlap_predictions(predictions, features, timeseries_length, hop_length)
