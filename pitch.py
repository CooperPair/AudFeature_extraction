#! /usr/bin/env python

# file for finding other features : 
'''
feature that is being extracted
1. min_pitch
2. max_pitch
3. mean_pitch
4. num_voice_breaks
5. percentage_breaks
6. speak_rate
7. num_pause
8. Total_dur_pause
9. no. of rise
10. no. of fall 
'''
import sys
import wave
import contextlib
from aubio import source, pitch
import numpy as np
import speech_recognition as sr
import os 

# for ignoring any warnings of required
import warnings
warnings.simplefilter("ignore")

if len(sys.argv) < 2:
    print("Usage: %s <filename> [samplerate]" % sys.argv[0])
    sys.exit(1)

'''
# for iterating over all the audio file
directory = 'audio/'

for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        print(filename)
    else:
        continue
'''

filename = sys.argv[1]

# calculating pitch
downsample = 1
samplerate = 44100 // downsample
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

win_s = 4096 // downsample # fft size
hop_s = 512  // downsample # hop size

s = source(filename, samplerate, hop_s)
samplerate = s.samplerate

tolerance = 0.8

pitch_o = pitch("yin", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []
times = []

# total number of frames read
total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    #pitch = int(round(pitch))
    confidence = pitch_o.get_confidence()
    #if confidence < 0.8: pitch = 0.
    #print("frames pitch confidence\n")
    time = total_frames / float(samplerate)
    print("%f %f %f" % (time, pitch, confidence))
    pitches += [pitch]
    times += [time]
    confidences += [confidence]
    total_frames += read
    if read < hop_s: break

if 0: sys.exit(0)

# converting pitch value to numpy value
pitches_np = np.array(pitches)
times_np = np.array(times)
print(pitches_np)
# print(times_np)


# code for finding number of voice breaks and percentage breaks and speak rates
# logic used is that when pitch changes to zero there will be voice breaks
count = 0
for i in range(len(pitches_np)-1):
    if pitches_np[i] == 0 and pitches_np[i] != pitches_np[i+1]:
        count += 1
    elif pitches_np[i] != 0 and pitches_np[i+1] == 0:
        count += 1
    else:
        continue

print(count)
per_breaks = (count/len(pitches_np))*100
print(str(per_breaks)+"%")


# finding min, max and mean pitch
pitches_np_max = np.max(pitches_np)
print(pitches_np_max)
pitches_np_mean = np.mean(pitches_np)
print(pitches_np_mean)
pitches_np_min = np.min(pitches_np)
print(pitches_np_min)

# finding total duration of pause that is when the pitch of the file is 0
# num of pause, max dur pause, average pause(wrt when it starts again), total duration of pause
pause_time = 0
for i in range(len(pitches_np)-1):
    if pitches_np[i] == 0:
        continue
    else:
        pause_time = times_np[i-1]

print(pause_time)

# finding total suration of file
with contextlib.closing(wave.open(filename, 'r')) as f :
    frame = f.getnframes()
    frame_float = float(frame)
    rate = f.getframerate()

    duration = frame / float(rate)
    print(duration)

# for finding the word in the file
r = sr.Recognizer()
audio_sample = sr.AudioFile(sys.argv[1])

with audio_sample as source:
    audio = r.record(source)

text_speech = r.recognize_google(audio)

play_time = duration - pause_time
speak_rate = len(text_speech) / play_time
print(speak_rate)

# rise and fall of the voice logic is that when pitch value is increasig there will be an increase in the voice else fall
num_rise = 0
num_fall = 0

for i in range(len(pitches_np)-1):
    if pitches_np[i] < pitches_np[i+1] and pitches_np[i+1] > pitches_np[i+2]:
        num_rise += 1
    else:
        continue

for i in range(len(pitches_np)-1):
    if pitches_np[i] > pitches_np[i+1] and pitches_np[i+1] < pitches_np[i+2]:
        num_fall += 1
    else:
        continue

print(num_rise)
print(num_fall)

# finding maximum duration of rise and fall of the voice in terms of time wrt pitch
# getting problem in dealing with logic 






# analysing graphically the pitches 
import os.path
from numpy import array, ma
import matplotlib.pyplot as plt
from demo_waveform_plot import get_waveform_plot, set_xlabels_sample2time

skip = 1

pitches = array(pitches[skip:])
confidences = array(confidences[skip:])
times = [t * hop_s for t in range(len(pitches))]

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1 = get_waveform_plot(filename, samplerate = samplerate, block_size = hop_s, ax = ax1)
plt.setp(ax1.get_xticklabels(), visible = False)
ax1.set_xlabel('')

def array_from_text_file(filename, dtype = 'float'):
    filename = os.path.join(os.path.dirname(__file__), filename)
    return array([line.split() for line in open(filename).readlines()],
        dtype = dtype)

ax2 = fig.add_subplot(312, sharex = ax1)
ground_truth = os.path.splitext(filename)[0] + '.f0.Corrected'
if os.path.isfile(ground_truth):
    ground_truth = array_from_text_file(ground_truth)
    true_freqs = ground_truth[:,2]
    true_freqs = ma.masked_where(true_freqs < 2, true_freqs)
    true_times = float(samplerate) * ground_truth[:,0]
    ax2.plot(true_times, true_freqs, 'r')
    ax2.axis( ymin = 0.9 * true_freqs.min(), ymax = 1.1 * true_freqs.max() )
# plot raw pitches
ax2.plot(times, pitches, '.g')
# plot cleaned up pitches
cleaned_pitches = pitches
#cleaned_pitches = ma.masked_where(cleaned_pitches < 0, cleaned_pitches)
#cleaned_pitches = ma.masked_where(cleaned_pitches > 120, cleaned_pitches)
cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches)
ax2.plot(times, cleaned_pitches, '.-')
#ax2.axis( ymin = 0.9 * cleaned_pitches.min(), ymax = 1.1 * cleaned_pitches.max() )
#ax2.axis( ymin = 55, ymax = 70 )
plt.setp(ax2.get_xticklabels(), visible = False)
ax2.set_ylabel('f0 (midi)')

# plot confidence
ax3 = fig.add_subplot(313, sharex = ax1)
# plot the confidence
ax3.plot(times, confidences)
# draw a line at tolerance
ax3.plot(times, [tolerance]*len(confidences))
ax3.axis( xmin = times[0], xmax = times[-1])
ax3.set_ylabel('condidence')
set_xlabels_sample2time(ax3, times[-1], samplerate)
plt.show()
# for saving the graph
#plt.savefig(os.path.basename(filename) + '.svg')