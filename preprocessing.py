from scipy import signal
import numpy as np

def subsampling (sig) :
    return signal.resample (sig, 1500)

def prep_data (func) :
   """
   @brief      decorator to use preprocessing functions returning one array with
               data matrix

   @param      function and signal

   @return     the prepped-preprocessed signal
   """
   def wrapper (fsig) :
        fsig [:,1] = func (fsig [:,1])
        fsig [:,2] = func (fsig [:,2])
        return fsig [:]
   return wrapper

@prep_data
def bwfilter (sig) :
   """
   @brief      third-order band-pass butterworth filter with cutoff frequencies
               of 0.4 and 45 Hz

   @param      a signal to filter

   @return     filtered signal
   """
   sos = signal.butter (3, [0.4, 45], 'bp', fs=360, output='sos')
   return signal.sosfilt (sos, sig )
