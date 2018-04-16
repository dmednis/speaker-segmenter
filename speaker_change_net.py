from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from dataset_loader import load_dataset

train_X, train_Y, test_X, test_Y = load_dataset()

callbacks = [
    TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=35),
    ModelCheckpoint("./models/model.{epoch:02d}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                    mode='auto', period=1)
]

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 35
nb_epochs = 3

print("Training X shape: " + str(train_X.shape))
print("Training Y shape: " + str(train_Y.shape))
print("Test X shape: " + str(test_X.shape))
print("Test Y shape: " + str(test_Y.shape))

input_shape = (train_X.shape[1], train_X.shape[2])
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=train_Y.shape[1], activation='softmax'))

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(train_X, train_Y, batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks)

print("\nTesting ...")
score, accuracy = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# 1 epoch
# Test loss:   0.056034792951317434
# Test accuracy:   0.980385290135853

# 35 epoch
# Test loss:   0.01425005547713424
# Test accuracy:   0.9953298309847269
