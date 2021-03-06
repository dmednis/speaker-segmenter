from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
from matplotlib import pyplot

from utils import ensure_dirs
from VADSequence import VADSequence
from dataset_loader import vad_voice_train, vad_noise_train, vad_voice_test, vad_noise_test

model_name = "vad2"
run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

opt = Adam()

batch_size = 100
timeseries_length = 100
nb_epochs = 10

voice_train = vad_voice_train()
noise_train = vad_noise_train()

voice_test = vad_voice_test()
noise_test = vad_noise_test()

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

print('Building LSTM RNN model ...')
input_shape = train_sample[0].shape

model = Sequential()

model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape[1:]))

model.add(Bidirectional(LSTM(32, return_sequences=True)))

model.add(TimeDistributed(Dense(40, activation="tanh")))

model.add(TimeDistributed(Dense(10, activation="tanh")))

model.add(TimeDistributed(Dense(1, activation="sigmoid")))

print("Compiling ...")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs/' + model_name + '-' + run, histogram_freq=0, batch_size=batch_size),
    ModelCheckpoint("./models/" + model_name + "_" + run + "/model_vad2.{epoch:02d}.hdf5", monitor='val_loss', verbose=0,
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
