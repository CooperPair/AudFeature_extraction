# importing necessary library
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn

audio_path = 'audio/human.wav'

# x = 1D array time series, sr is sampling rate of x by default 22KHz
x, sr = librosa.load(audio_path)
# print(type(x), type(sr))

# display spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
plt.title("Spectrogram")
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
plt.colorbar()
plt.show()

# extracting features
# ZCR
plt.figure(figsize=(14,5))
plt.title("Zero Cross Rating")
librosa.display.waveplot(x, sr=sr)
plt.show()

# Zooming in
p1 = 9000
p2 = 9100
plt.figure(figsize=(14,5))
plt.plot(x[p1:p2])
plt.title("Zoomed in view")
plt.grid()
plt.show()

# spectral centroid
zero_crossings = librosa.zero_crossings(x[p1:p2], pad=False)
print("The total sum of the zero crossing value is :"+str(sum(zero_crossings)))

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title("Spectral Centroid")
plt.show()

# spectral roll off
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title("Spectral Rolloff")
plt.show()

# mfcc plot
mfccs = librosa.feature.mfcc(x, sr=sr)
print("Shape of MFCC is"+str(mfccs.shape))
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title("MFCC")
plt.show()