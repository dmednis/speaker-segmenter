from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np

from VADSequence import VADSequence
from dataset_loader import load_dataset

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=35),
    ModelCheckpoint("./models/model.{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False,
                    save_weights_only=False,
                    mode='auto', period=1)
]

opt = Adam()

batch_size = 30
timeseries_length = 1
nb_epochs = 3

voice_train_x, _, voice_test_x, _ = load_dataset()

noise = np.load("./noise/noise_x.npy")

print(noise.shape)
print(noise[0].shape)
print(voice_train_x.shape)
print(voice_test_x.shape)
exit(0)
train_generator = VADSequence(voice_train_x, noise[:], timeseries_length=timeseries_length, batch_size=batch_size)
test_generator = VADSequence(test_x, test_y, timeseries_length=timeseries_length, batch_size=batch_size)

train_sample = train_generator[0]
test_sample = test_generator[0]
print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
print("Test X shape: " + str(test_sample[0].shape))
print("Test Y shape: " + str(test_sample[1].shape))

print('Building CONV LSTM RNN model ...')
input_shape = train_sample[0].shape

print(input_shape)

filters = 32
kernel_size = 3
pool_size = 2

model = Sequential()

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=input_shape[1:]))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(LSTM(32))

model.add(Dense(2, activation="softmax"))

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit_generator(train_generator, epochs=nb_epochs, callbacks=callbacks)

print("\nTesting ...")
score, accuracy = model.evaluate_generator(test_generator, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# 1 epoch
