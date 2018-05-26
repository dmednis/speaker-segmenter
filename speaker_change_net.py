from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, LeakyReLU, Dropout, BatchNormalization, TimeDistributed, ZeroPadding1D, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
from matplotlib import pyplot

from utils import ensure_dirs
from SEGSequence import SEGSequence
from dataset_loader import seg_speakers_train, seg_speakers_test

model_name = "seg"
run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
gpu = True

opt = Adam(lr=0.001, decay=0.00005)

batch_size = 100
timeseries_length = 100
nb_epochs = 1000

speakers_train = seg_speakers_train()
print("Speakers train set shape", speakers_train.shape)
train_generator = SEGSequence(speakers_train,
                              timeseries_length=timeseries_length,
                              batch_size=batch_size,
                              name="train")
del speakers_train

speakers_test = seg_speakers_test()
print("Speakers test set shape", speakers_test.shape)
test_generator = SEGSequence(speakers_test,
                             timeseries_length=timeseries_length,
                             batch_size=batch_size,
                             name="test")
del speakers_test

train_sample = train_generator[0]
test_sample = test_generator[0]
print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
print("Test X shape: " + str(test_sample[0].shape))
print("Test Y shape: " + str(test_sample[1].shape))

print('Building CONV LSTM RNN model ...')
input_shape = train_sample[0].shape

print(input_shape)

recurrent_layer = CuDNNLSTM if gpu else LSTM

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

model.add(TimeDistributed(Dense(40)))

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
    ModelCheckpoint("./models/" + model_name + "_" + run + "/model_seg.{epoch:02d}.hdf5", monitor='val_loss',
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
                              callbacks=callbacks,)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
