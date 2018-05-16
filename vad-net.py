from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, TimeDistributed, ZeroPadding1D
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
from matplotlib import pyplot

from utils import train_test_split, ensure_dirs, shuffle
from VADSequence import VADSequence
from dataset_loader import vad_voice_train, vad_noise_train, vad_voice_test, vad_noise_test

run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

ensure_dirs(["./models", "./models/" + run])

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

batch_size = 20
timeseries_length = 44
nb_epochs = 15

voice_train = vad_voice_train()
noise_train = vad_noise_train()

voice_test = vad_voice_test()
noise_test = vad_noise_test()

ratio = voice_train.shape[0] / noise_train.shape[0]

print("Noise train set shape", noise_train.shape)
print("Noise test set shape", noise_test.shape)
print("Voice train set shape", voice_train.shape)
print("Voice test set shape", voice_test.shape)

train_generator = VADSequence(voice_train, noise_train, timeseries_length=timeseries_length, batch_size=batch_size)
test_generator = VADSequence(voice_test, noise_test, timeseries_length=timeseries_length, batch_size=batch_size)

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

model.add(ZeroPadding1D(1,
                        input_shape=input_shape[1:]))

model.add(Conv1D(128,
                 3,
                 padding='valid',
                 activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.6))

model.add(GRU(128, return_sequences=True))

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(GRU(128, return_sequences=True))

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(TimeDistributed(Dense(1, activation="sigmoid")))

print("Compiling ...")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs/' + run, histogram_freq=0, batch_size=batch_size),
    ModelCheckpoint("./models/" + run + "/model_vad.{epoch:02d}.hdf5", monitor='val_loss', verbose=0,
                    save_best_only=False,
                    save_weights_only=False,
                    mode='auto', period=1)
]

print("Training ...")
history = model.fit_generator(train_generator,
                              epochs=nb_epochs,
                              validation_data=test_generator,
                              callbacks=callbacks,
                              # class_weight={0: ratio,
                              #               1: 1.}
                              )

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
