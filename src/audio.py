# import librosa as lb
# import soundfile as sf
# import numpy as np
# import os, glob, pickle
 
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# from pydub import AudioSegment
# emotion_labels = {
#   '01':'neutral',
#   '02':'calm',
#   '03':'happy',
#   '04':'sad',
#   '05':'angry',
#   '06':'fearful',
#   '07':'disgust',
#   '08':'surprised'
# }

# focused_emotion_labels = ['happy', 'sad', 'angry']
# def audio_features(file_title, mfcc, chroma, mel):
    
#     sound = AudioSegment.from_wav(file_title)
#     sound = sound.set_channels(1)
#     sound.export("data1/"+file_title, format="wav")

#     with sf.SoundFile("data1/"+file_title) as audio_recording:
#         audio = audio_recording.read(dtype="float32")
#         sample_rate = audio_recording.samplerate
#         if chroma:
#             stft=np.abs(lb.stft(audio))
#             result=np.array([])
#         if mfcc:
#             mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
#             result=np.hstack((result, mfccs))
#         if chroma:
#             chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#             result=np.hstack((result, chroma))
#         if mel:
#             mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
#             result=np.hstack((result, mel))
#         return result


# def loading_audio_data(file):
#     x = []
#     y = []
    
    
#     file_path=os.path.basename(file)
#     emotion = 'angry'
    
    
#     feature = audio_features(file, mfcc=True, chroma=True, mel=True)
#     x.append(feature)
#     y.append(emotion)
#     return x, y

# audiomodel = pickle.load(open("audiomodel.h5", 'rb'))

# X, y = loading_audio_data("recorded8.wav")
# y_pred = audiomodel.predict(X)
# print(y_pred)
# # accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# # print("Accuracy of the Recognizer is: {:.1f}%".format(accuracy*100))