import pyaudio
import wave
import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle
 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
emotion_labels = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

focused_emotion_labels = ['happy', 'sad', 'angry']
def audio_features(file_title, mfcc, chroma, mel):
    
    sound = AudioSegment.from_wav(file_title)
    sound = sound.set_channels(1)
    sound.export("data1/"+file_title, format="wav")

    with sf.SoundFile("data1/"+file_title) as audio_recording:
        audio = audio_recording.read(dtype="float32")
        sample_rate = audio_recording.samplerate
        if chroma:
            stft=np.abs(lb.stft(audio))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result

def loading_audio_data(file):
    x = []    
    
    file_path=os.path.basename(file)

    feature = audio_features(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    return x
# the file name output you want to record into
filename = "recorded8.wav"
# set the chunk size of 1024 samples
chunk = 150
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100
record_seconds = 5
# initialize PyAudio object
p = pyaudio.PyAudio()
# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
frames = []
print("Recording...")
for i in range(int(44100 / chunk * record_seconds)):
    data = stream.read(chunk)
    # if you want to hear your voice while recording
    # stream.write(data)
    frames.append(data)
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()
# save audio file
# open the file in 'write bytes' mode
wf = wave.open(filename, "wb")
# set the channels
wf.setnchannels(1)
# set the sample format
wf.setsampwidth(p.get_sample_size(FORMAT))
# set the sample rate
wf.setframerate(sample_rate)
# write the frames as bytes
wf.writeframes(b"".join(frames))
# close the file
wf.close()


wf = wave.open(filename, "rb")
audiomodel = pickle.load(open("audiomodel.h5", 'rb'))

X = loading_audio_data(filename)
y_pred = audiomodel.predict(X)
print(y_pred)
