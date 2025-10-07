import crepe
import soundfile as sf

audio, sr = sf.read("test.wav")
time, frequency, confidence, _ = crepe.predict(audio, sr)

print(time[:5], frequency[:5], confidence[:5])