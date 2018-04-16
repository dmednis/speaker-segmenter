import os
import glob
import numpy as np
import librosa
from make_timit import extract_features

degradation_samples = ["ambience-pub.wav", "applause.wav",
                       "brown-noise.wav", "white-noise.wav"]

preloaded_degradations = []


def preload_degradations():
    global preloaded_degradations
    for degradation in degradation_samples:
        z, sr = librosa.core.load("./degradations/" + degradation, mono=True)
        preloaded_degradations.append(z)
    preloaded_degradations = extract_features(preloaded_degradations, 22050)


def load_dataset():
    train_X_filename = "./speakers/train_X.npy"
    train_Y_filename = "./speakers/train_Y.npy"
    test_X_filename = "./speakers/test_X.npy"
    test_Y_filename = "./speakers/test_Y.npy"

    if (os.path.isfile(train_X_filename) and
            os.path.isfile(train_Y_filename) and
            os.path.isfile(test_X_filename) and
            os.path.isfile(test_Y_filename)):
        print("Loading training data")
        train_X = np.load(train_X_filename)
        train_Y = np.load(train_Y_filename)
        print("Loading testing data")
        test_X = np.load(test_X_filename)
        test_Y = np.load(test_Y_filename)
        return train_X, train_Y, test_X, test_Y
    else:
        train_X, train_Y, test_X, test_Y = prepare_dataset()
        np.save(train_X_filename, train_X)
        np.save(train_Y_filename, train_Y)
        np.save(test_X_filename, test_X)
        np.save(test_Y_filename, test_Y)
        return train_X, train_Y, test_X, test_Y


def prepare_dataset():
    preload_degradations()

    train_X = np.ndarray((0, 128, 33))
    train_Y = np.ndarray((0, 4))
    test_X = np.ndarray((0, 128, 33))
    test_Y = np.ndarray((0, 4))

    print("Loading training data")
    for train_fragment in glob.iglob("./speakers/train/*.npy"):
        print("Loading training file ", train_fragment)
        data = np.load(train_fragment)
        
        train_X = np.concatenate((train_X, data[0]))
        train_Y = np.concatenate((train_Y, [[0, 0, 1, 0]], np.array([[1, 0, 0, 0]] * (len(data[0]) - 1))))

        train_X = np.concatenate((train_X, preloaded_degradations[0]))
        train_Y = np.concatenate(
            (train_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[0]) - 1))))

        train_X = np.concatenate((train_X, data[1]))
        train_Y = np.concatenate((train_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[1]) - 1))))

        train_X = np.concatenate((train_X, preloaded_degradations[1]))
        train_Y = np.concatenate(
            (train_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[1]) - 1))))

        train_X = np.concatenate((train_X, data[2]))
        train_Y = np.concatenate((train_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[2]) - 1))))

        train_X = np.concatenate((train_X, preloaded_degradations[2]))
        train_Y = np.concatenate(
            (train_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[2]) - 1))))

        train_X = np.concatenate((train_X, data[3]))
        train_Y = np.concatenate((train_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[3]) - 1))))

        train_X = np.concatenate((train_X, preloaded_degradations[3]))
        train_Y = np.concatenate(
            (train_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[3]) - 1))))

        train_X = np.concatenate((train_X, data[4]))
        train_Y = np.concatenate((train_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[4]) - 1))))

    print("Loading testing data")
    for test_fragment in glob.iglob("./speakers/test/*.npy"):
        print("Loading testing file ", test_fragment)
        data = np.load(test_fragment)
        test_X = np.concatenate((test_X, data[0]))
        test_Y = np.concatenate((test_Y, [[0, 0, 1, 0]], np.array([[1, 0, 0, 0]] * (len(data[0]) - 1))))

        test_X = np.concatenate((test_X, preloaded_degradations[0]))
        test_Y = np.concatenate(
            (test_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[0]) - 1))))

        test_X = np.concatenate((test_X, data[1]))
        test_Y = np.concatenate((test_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[1]) - 1))))

        test_X = np.concatenate((test_X, preloaded_degradations[2]))
        test_Y = np.concatenate(
            (test_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[2]) - 1))))

        test_X = np.concatenate((test_X, data[2]))
        test_Y = np.concatenate((test_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[2]) - 1))))

        test_X = np.concatenate((test_X, preloaded_degradations[1]))
        test_Y = np.concatenate(
            (test_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[1]) - 1))))

        test_X = np.concatenate((test_X, data[3]))
        test_Y = np.concatenate((test_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[3]) - 1))))

        test_X = np.concatenate((test_X, preloaded_degradations[3]))
        test_Y = np.concatenate(
            (test_Y, [[0, 1, 0, 0]], np.array([[1, 0, 0, 0]] * (len(preloaded_degradations[3]) - 1))))

        test_X = np.concatenate((test_X, data[4]))
        test_Y = np.concatenate((test_Y, [[0, 0, 0, 1]], np.array([[1, 0, 0, 0]] * (len(data[4]) - 1))))

    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    load_dataset()
