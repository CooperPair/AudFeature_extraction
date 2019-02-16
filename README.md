# AudFeature_extraction
An approach to extract most of the features from an audio file

### Createing virtual environment
 Run the following command in command shell
```
foo@baar python3 -m virtualenv audio
```
 Then activate the given virtual envirionement by running the following command in command shell :
```
foo@bar source env/bin/activate
```

### To install all the required packages in the given environment:
 Run the following command in command shell
```
foo@bar pip install -r requirement.txt
```
This will install all the necessary packages that is required to extract the features from an audio.

### pitch.py file

This file conatains the code that can find the pitch value with the help of a library **aubio** in python.
Then with the help of pitch we can find other features like voice breaks, max/min/mean pitch value and many more see
wiki for details.
In order to see the result run the following command
```
foo@bar python pitch.py /audio/human.wav
```

### energy.py

This file contain the code for finding the energy value and its representation in terms of band.
It also uses the same library **aubio** as above.
In order to see the result run the following command
```
foo@bar python energy.py /audio/human.wav
```
This will display the energy value at each regular interval as well as the energy band as a graph.

### spectrogram.py

This file is used to represent the audio in different format like spectrogram, spectrogram roll off, spectrogram centroid,
mfcc etc.It uses the library **librosa** in python
See the result by running this code
```
foo@bar python spectrogram.py /audio/human.wav
```
The importnace of spectrogram is that it can easily be used as an input feature to any neural networl which can be used to extract some important features.

### extra_feature_extract.py

This file contain the code that can be used to extract some of the measure and important features from the audio file.
In order to see the result from this file run the following command in command shell
```
foo@bar python extra_feature_extract.py human.wav
```
