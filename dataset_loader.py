import os
import glob
import numpy as np


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
    train_X = np.ndarray((0, 128, 33))
    train_Y = np.ndarray((0, 2))
    test_X = np.ndarray((0, 128, 33))
    test_Y = np.ndarray((0, 2))

    print("Loading training data")
    for train_fragment in glob.iglob("./speakers/train/*.npy"):
        print("Loading training file ", train_fragment)
        data = np.load(train_fragment)
        train_X = np.concatenate((train_X, data))
        train_Y = np.concatenate((train_Y, [[1, 0]], np.array([[0, 1]] * (len(data) - 1))))

    print("Loading testing data")
    for test_fragment in glob.iglob("./speakers/test/*.npy"):
        print("Loading testing file ", test_fragment)
        data = np.load(test_fragment)
        test_X = np.concatenate((test_X, data))
        test_Y = np.concatenate((test_Y, [[1, 0]], np.array([[0, 1]] * (len(data) - 1))))

    return train_X, train_Y, test_X, test_Y
