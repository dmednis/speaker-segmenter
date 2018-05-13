from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from matplotlib import pyplot
from VADSequence import VADSequence
from dataset_loader import load_dataset

opt = Adam()

batch_size = 20
timeseries_length = 44
nb_epochs = 5

voice_train_x, _, voice_test_x, _ = load_dataset()

noise = np.load("./noise/noise_x_flat.npy")
noise_test = noise[:700000]
noise_train = noise[700000:]

print("Noise train set shape", noise_train.shape)
print("Noise test set shape", noise_test.shape)
print("Voice train set shape", voice_train_x.shape)
print("Voice test set shape", voice_test_x.shape)

train_generator = VADSequence(voice_train_x, noise_train, timeseries_length=timeseries_length, batch_size=batch_size)
test_generator = VADSequence(voice_test_x, noise_test, timeseries_length=timeseries_length, batch_size=batch_size)

train_sample = train_generator[0]
test_sample = test_generator[0]
print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
print("Test X shape: " + str(test_sample[0].shape))
print("Test Y shape: " + str(test_sample[1].shape))

print('Building CONV LSTM RNN model ...')
input_shape = train_sample[0].shape

print(input_shape)

# filters = 32
# kernel_size = 3
# pool_size = 2

model = Sequential()

model.add(Conv1D(32,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=input_shape[1:]))

# model.add(Conv1D(64,
#                  3,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))

# model.add(Conv1D(128,
#                  3,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
#
# model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.5))

model.add(LSTM(64))

# model.add(LSTM(256))
#
# model.add(Dropout(0.4))
#
# model.add(Dense(128, activation="relu"))
#
# model.add(Dropout(0.3))
#
# model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

# model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

print("Compiling ...")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=batch_size),
    ModelCheckpoint("./models/model_vad.{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False,
                    save_weights_only=False,
                    mode='auto', period=1)
]

print("Training ...")
history = model.fit_generator(train_generator,
                              epochs=nb_epochs,
                              validation_data=test_generator,
                              callbacks=callbacks)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

# print("\nTesting ...")
# score, accuracy = model.evaluate_generator(test_generator)
# print("Test loss:  ", score)
# print("Test accuracy:  ", accuracy)

# 1 epoch
