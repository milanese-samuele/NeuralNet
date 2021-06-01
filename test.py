## Collection of tests for development
import numpy as np
from utils import *
from preprocessing import *

FS = 360 # sample frequency per second

def test_filter () :
    full = readcsv ("./mitbih_database/201.csv")

    full = bwfilter (full)

    wins = createwindows (full [FS:FS*60], 4, [FS,FS*60])
    windowplots (wins)
