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
        axis[i].plot  (current [:,0], current [:,1], label = 'MLII')
        axis[i].plot  (current [:,0], current [:,2], label = 'V5')
    plt.legend ()
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

def annotations_to_signal(labels):
    categories = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', '[', '!', ']', 'e', 'j', 'E', '/', 'f', 'x', 'Q', '|']
    tsignals = []
    for l in labels:
        s = [0] * (len(categories) + 1)
        s[categories.index(l) if l in categories else len(categories)] = 1
        tsignals.append(s)
    return tsignals

def extract_annotations (pnumber: int, data_path="./mitbih_database/"):
    mat = np.loadtxt (data_path + str(pnumber) + "annotations.txt", dtype=str, skiprows=1, usecols = (1,2))
    ts = [int (nstr) for nstr in mat[:,0]]
    sigs = annotations_to_signal(mat[:,1])
    ret = np.asarray([np.asarray([ts[i], sigs[i]]) for i in range(len(ts))])
    return ret
