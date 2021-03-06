from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Dense, Conv1D, LeakyReLU, Dropout, BatchNormalization, TimeDistributed, ZeroPadding1D, CuDNNGRU
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
from matplotlib import pyplot

from utils import ensure_dirs
from VADSequence import VADSequence
from dataset_loader import vad_voice_train, vad_noise_train, vad_voice_test, vad_noise_test

model_name = "vad2"
run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
gpu = True

ensure_dirs(["./models", "./models/" + run])

opt = Adam(lr=0.0001, decay=0.00005)

batch_size = 100
timeseries_length = 100
nb_epochs = 60

voice_train = vad_voice_train()
noise_train = vad_noise_train()
print("Voice train set shape", voice_train.shape)
print("Noise train set shape", noise_train.shape)
train_generator = VADSequence(voice_train,
                              noise_train,
                              timeseries_length=timeseries_length,
                              batch_size=batch_size,
                              name="train")
del voice_train
del noise_train

voice_test = vad_voice_test()
noise_test = vad_noise_test()
print("Voice test set shape", voice_test.shape)
print("Noise test set shape", noise_test.shape)
test_generator = VADSequence(voice_test, noise_test,
                             timeseries_length=timeseries_length,
                             batch_size=batch_size,
                             name="test")
del voice_test
del noise_test


train_sample = train_generator[0]
test_sample = test_generator[0]
print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
print("Test X shape: " + str(test_sample[0].shape))
print("Test Y shape: " + str(test_sample[1].shape))

print('Building CONV LSTM RNN model ...')
input_shape = train_sample[0].shape

print(input_shape)

recurrent_layer = CuDNNGRU if gpu else GRU

model = Sequential()

model.add(ZeroPadding1D(1,
                        input_shape=input_shape[1:]))

model.add(Conv1D(128, 3))

model.add(LeakyReLU())

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(ZeroPadding1D(1))

model.add(Conv1D(64, 3))

model.add(LeakyReLU())

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(recurrent_layer(64, return_sequences=True))

model.add(LeakyReLU())

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(recurrent_layer(32, return_sequences=True))

model.add(LeakyReLU())

model.add(Dropout(0.4))

model.add(TimeDistributed(Dense(10)))

model.add(LeakyReLU())

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(TimeDistributed(Dense(1, activation="sigmoid")))

print("Compiling ...")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs/' + model_name + '-' + run, histogram_freq=0, batch_size=batch_size),
    ModelCheckpoint("./models/" + model_name + "_" + run + "/model_vad2.{epoch:02d}.hdf5", monitor='val_loss',
                    verbose=0,
                    save_best_only=False,
                    save_weights_only=False,
                    mode='auto', period=1)
]

print("Training ...")
ensure_dirs(["./models", "./models/" + model_name + "_" + run])

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


def model(input_shape, optimizer):
    print('Building CONV LSTM RNN model ...')

    recurrent_layer = CuDNNGRU if gpu else GRU

    model = Sequential()

    model.add(ZeroPadding1D(1,
                            input_shape=input_shape))

    model.add(Conv1D(128, 3))

    model.add(LeakyReLU())

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(ZeroPadding1D(1))

    model.add(Conv1D(64, 3))

    model.add(LeakyReLU())

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(recurrent_layer(64, return_sequences=True))

    model.add(LeakyReLU())

    model.add(Dropout(0.4))

    model.add(BatchNormalization())

    model.add(recurrent_layer(32, return_sequences=True))

    model.add(LeakyReLU())

    model.add(Dropout(0.4))

    model.add(TimeDistributed(Dense(10)))

    model.add(LeakyReLU())

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(TimeDistributed(Dense(1, activation="sigmoid")))

    print("Compiling ...")
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model
