## Collection of tests for development
import numpy as np
from rnn import RNN
from utils import *
from preprocessing import *

FS = 360 # sample frequency per second

def test_filter () :
    full = readcsv ("./mitbih_database/201.csv")

    full = bwfilter (full)

    wins = createwindows (full [FS:FS*60], 4, [FS,FS*60])
    windowplots (wins)

def test_arch () :
    full = readcsv ("./mitbih_database/201.csv")
    full = bwfilter (full)
    nots = full [720:3600, 1:3]
    # plt.plot (range (len (nots)), nots )
    # plt.show ()
    x = RNN ()
    x.test_run (nots)
