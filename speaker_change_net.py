from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed, Conv2D, ConvLSTM2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from SpeakerSequence import SpeakerSequence

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=35),
    ModelCheckpoint("./models/model.{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False,
                    save_weights_only=False,
                    mode='auto', period=1)
]

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 1
nb_epochs = 3

train_generator = SpeakerSequence("speakers/train_o", batch_size)
test_generator = SpeakerSequence("speakers/test_o", batch_size)

train_sample = train_generator.__getitem__(0)
# test_sample = test_generator.__getitem__(0)

print("Training X shape: " + str(train_sample[0].shape))
print("Training Y shape: " + str(train_sample[1].shape))
# print("Test X shape: " + str(test_sample[0].shape))
# print("Test Y shape: " + str(test_sample[1].shape))

print('Build LSTM RNN model ...')
model = Sequential()
model.add(Conv2D(32, (3, 3), data_format="channels_last", input_shape=(85, 1025, 1)))
model.summary()
model.add(ConvLSTM2D())
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=4, activation='softmax'))

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit_generator(train_generator, batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks)

print("\nTesting ...")
score, accuracy = model.evaluate_generator(test_generator, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# 1 epoch
# Test loss:   0.056034792951317434
# Test accuracy:   0.980385290135853

# 35 epoch
# Test loss:   0.01425005547713424
# Test accuracy:   0.9953298309847269
