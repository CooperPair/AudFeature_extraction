
'''
Feautres that are being extracted : 
Zero cross Rate
Energy
Entropy of Energy
Spectral Centroid
Spectral Spread
Spectral Entropy
Spectral Flux
Spectral Roll off
MFCC
Chroma Vector
Chroma Deviation
'''

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
import sys

import warnings
warnings.simplefilter("ignore")

[Fs, x] = audioBasicIO.readAudioFile("audio/" + sys.argv[1])
# Fs is the frame rate in the audio signal and x is the numpy array represennting the sample data.

if( len( x.shape ) > 1 and  x.shape[1] == 2 ):
    x = np.mean( x, axis = 1, keepdims = True )
else:
    x = x.reshape( x.shape[0], 1 )

F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, 0.050*Fs, 0.025*Fs)
print(f_names) # features name
print(F) # features values

plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no')
plt.ylabel(f_names[0]) 
plt.subplot(2,1,2); plt.plot(F[1,:])
plt.xlabel('Frame no')
plt.ylabel(f_names[1])
plt.show()