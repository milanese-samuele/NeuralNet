import numpy as np
from utils import *
from preprocessing import *
import matplotlib.pyplot as plt

def main():

    full = readcsv ("./mitbih_database/100.csv")

    full [:,1] = bwfilter (full [:,1])
    full [:,2] = bwfilter (full [:,2])

    wins = createwindows (full [:21600], 4, [0,21600])
    windowplots (wins)
    # plt.plot (np.linspace(0, 1, len (filtered), False), filtered  )
    # plt.show ()

if __name__ == '__main__':
    main()
