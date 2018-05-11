from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

from SpeechSequence import VADSequence

data = np.array(range(100), dtype=np.int8)

print(data.shape)
data = np.array_split(data, 21)
print(data)
data = np.array(data)
print(data.shape)

print(100 / 6)