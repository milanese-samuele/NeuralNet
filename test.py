## Collection of tests for development
import numpy as np
from rnn import RNN
from utils import *
from preprocessing import *

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

    wins = preprocess(201)
    print(wins[0])
    print(len(wins))

    ## windows = 44 samples before r-peak and 70 before r-peak
    for win in wins:
        plt.plot(win[0], win[1])
        plt.show()

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
