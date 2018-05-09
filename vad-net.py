from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, InputLayer, Reshape, ZeroPadding2D, \
    Dropout, Activation, ConvLSTM2D, BatchNormalization, Conv3D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from SpeechSequence import SpeechSequence
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

train_x, train_y, test_x, test_y = load_dataset()

train_generator = SpeechSequence(train_x, train_y, timeseries_length=timeseries_length, batch_size=batch_size)
test_generator = SpeechSequence(test_x, test_y, timeseries_length=timeseries_length, batch_size=batch_size)

train_sample = train_generator[0]
test_sample = test_generator[0]
print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
print("Test X shape: " + str(test_sample[0].shape))
print("Test Y shape: " + str(test_sample[1].shape))

print('Building CONV LSTM RNN model ...')
input_shape = train_sample[0].shape[2:]
num_chan = 1

model = Sequential()

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     input_shape=(None,) + input_shape,
                     padding='same', return_sequences=True, data_format="channels_first"))


model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     padding='same', return_sequences=True, data_format="channels_first"))


model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     padding='same', return_sequences=True, data_format="channels_first"))


model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     padding='same', return_sequences=False, data_format="channels_first"))
model.add(Flatten())

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
