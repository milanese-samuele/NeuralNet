import numpy as np
from utils import *
from preprocessing import *
import matplotlib.pyplot as plt

def main():

    full = readcsv ("./mitbih_database/100.csv")
    # wins = createwindows (full, 10, [0,1000])
    # windowplots (wins)

    filtered = bwfilter (full [:,1])
    sub = subsampling (filtered)
    plt.plot (np.linspace(0, 1, 150, False), sub [:150] )
    plt.show ()

if __name__ == '__main__':
    main()
