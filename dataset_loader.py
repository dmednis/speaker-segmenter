import os
import glob
import numpy as np


def load_dataset():
    train_x_filename = "./speakers/train_X.npy"
    train_y_filename = "./speakers/train_Y.npy"
    test_x_filename = "./speakers/test_X.npy"
    test_y_filename = "./speakers/test_Y.npy"

    if (os.path.isfile(train_x_filename) and
            os.path.isfile(train_y_filename) and
            os.path.isfile(test_x_filename) and
            os.path.isfile(test_y_filename)):
        print("Loading training data")
        train_x = np.load(train_x_filename)
        train_y = np.load(train_y_filename)
        print("Loading testing data")
        test_x = np.load(test_x_filename)
        test_y = np.load(test_y_filename)
        return train_x, train_y, test_x, test_y
    else:
        train_x, train_y = prepare_dataset("train")
        test_x, test_y = prepare_dataset("test")
        np.save(train_x_filename, train_x)
        np.save(train_y_filename, train_y)
        np.save(test_x_filename, test_x)
        np.save(test_y_filename, test_y)
        return train_x, train_y, test_x, test_y


def add_labels(x):
    y = []
    for i in range(x.shape[0]):
        y.append([1, 0])
    y = np.array(y)
    return y


def prepare_dataset(dataset):
    print("Loading " + dataset + " data")
    speakers = glob.glob("./speakers/" + dataset + "/*.npy")
    all_speakers = []
    for speaker in speakers:
        data = np.load(speaker)
        all_speakers.append(data)
    x = np.concatenate(all_speakers)
    y = add_labels(x)
    return x, y


if __name__ == "__main__":
    load_dataset()
