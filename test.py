## Collection of tests for development
import numpy as np
from rnn import RNN
from utils import *
from preprocessing import *
from sigs import Window
from patientclass import Patient


pns = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
       111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
       122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
       209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
       222, 223, 228, 230, 231, 232, 233, 234]

FS = 360 # sample frequency per second

def test_filter () :
    full = readcsv ("./mitbih_database/201.csv")

    full = bwfilter (full)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(full)
    anns = np.asarray (extractAnnotations ("./mitbih_database/201annotations.txt"))


    bounds = create_boundaries (anns)

    windows = split_mat (full, bounds)

    i = 0
    plen = 0
    for win in windows :
        mark =anns [i]
        plt.plot (win [:,0], win [:,1])
        plt.plot (mark, win [(mark - plen),1], 'ro')
        plt.show ()
        plen += len (win)
        i+=1
    # wins = createwindows (full [FS:FS*60], 4, [FS,FS*60])
    # windowplots (wins)
    # In ("./mitbih_database/200.csv")

def test_pam () :
    tmp = window_factory(201)
    anormal = [win for win in tmp if win.btype != 'N']
    for win in anormal:
        plt.plot (range (len (win.signal)), win.signal)
        plt.title (win.btype)
        plt.show ()

def find_table():
    for p in pns:
        tmp = window_factory (p)
        labs = [w.btype for w in tmp]
        print ("patient: {}\t{}".format (p, set (labs)))

def create_artifacts ():
    for i in pns:
        Patient (i)

def occurrence_table ():
    # c = 0
    # for p in pns:
    #     c += list (extract_annotations (p) [:,1]).count ('L')
    # print (c)
    ul = []
    for p in [Patient (i) for i in pns]:
        l = set ([w.btype for w in p.wins])
        b = ('A' in l) + ('N' in l) + ('R' in l) + ('L' in l) + ('V' in l)
        ul.append (np.asarray ([p, l,  b]))
    acc=[]
    ul = np.asarray ([x for x in ul if x [2]>=3])
    for p in ul[:,0]:
        l = [w.btype for w in p.wins]
        for x in l:
            acc.append (x)
    print (len (ul))
    print (Sort_Tuple ([[x, acc.count (x)] for x in set (acc)],1 ))

# borrowed by geeksforgeeks
def Sort_Tuple(tup, pos):

    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst-i-1):
            if (tup[j][pos] > tup[j + 1][pos]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup


def test_arch () :
    full = readcsv ("./mitbih_database/201.csv")
    full = bwfilter (full)
    nots = full [720:3600, 1:3]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(nots)
    plt.plot (range (len (data_scaled)), data_scaled )
    plt.show ()
    # x = RNN ()
    # print (x.Whid)
    # x.test_run (data_scaled)
