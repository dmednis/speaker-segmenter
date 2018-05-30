import librosa

file = "./samples/seg-test"

y, sr = librosa.load(file + ".wav", sr=16000, mono=True)
librosa.output.write_wav(file + "16.wav", y, sr)
