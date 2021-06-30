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

def annotations_to_signal(labels, categories):
    tsignals = []
    for l in labels:
        s = [0] * (len(categories))
        s[categories.index(l) if l in categories else len(categories)] = 1
        tsignals.append(np.asarray (s))
    return tsignals

def match_beat_type(sig, symbol: str) -> bool:
    categories = ['N', 'L', 'R', 'A', 'a',
                  'J', 'S', 'V', 'F', '[',
                  '!', ']', 'e', 'j', 'E',
                  '/', 'f', 'x', 'Q', '|']
    return (sig.index(1) == categories.index(symbol))

## ?? DONT REMEMBER
# def assign_annotations(windows, anns):
#     for sample, btype in anns:
#         for i in range(len(windows)):
#             if sample in windows[i, 0]:
#                 windows[i] = np.append(windows[i], btype)
#                 break
#     return windows


def extract_annotations (pnumber: int, data_path="./mitbih_database/"):
    mat = np.loadtxt (data_path + str(pnumber) + "annotations.txt", dtype=str, skiprows=1, usecols = (1,2))
    return np.asarray ([[int (nstr), a] for nstr, a in mat], dtype = object)
