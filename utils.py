import csv
import numpy as np
import matplotlib.pyplot as plt

def readcsv (filename : str) :
    """
    @param      a string filename for extraction of database

    @return     return a np matrix of data
    """
    reader = csv.reader (open (filename))
    fields = next (reader)
    x = list (reader)
    return np.array (x).astype (int)

def windowplots (windows : list) :
    """
    @param      a list of window signals
    """
    figure, axis = plt.subplots (len (windows),1)

    for i in range (len (windows)):
        current = np.array (windows [i]).astype (int)
        axis[i].plot  (current [:,0], current [:,1], current [:,2])
    plt.show ()

def createwindows (mat, n : int, interval : tuple):
   """
   @param      matrix of data, number of windows, the interval of the data

   @return     a list of windows
   """
   slice = mat [interval [0]: interval [1]]
   step = int (len (slice)/n)
   windows = []
   for i in np.arange (0, len (slice), step):
       windows.append (slice [i:i+step-1])
   return windows
