import numpy as np
from ecgdetectors import Detectors
import matplotlib.pyplot as plt
from utils import *
from preprocessing import *
from patientclass import Patient

def show_window ():
    wins = Patient (108).wins
    n = 190
    sig = wins [n].signal
    lab = wins [n].btype
    plt.plot (range (len (sig)), sig)
    plt.title (f"Window #190 from Patient 108 Type: Normal", fontsize = 22)
    plt.xlabel ("# Sample", fontsize = 18)
    plt.ylabel ("Sample Value", fontsize = 18)
    plt.show ()


def show_filter():
    full = readcsv("./mitbih_database/201.csv")
    # full = bwfilter (full)
    # time = np.linspace (60,90, num=1800)
    # plt.plot (time ,full [3600:5400,1], label="MLII lead")
    # plt.plot (time ,full [3600:5400,2], label="V1 lead")
    # plt.xlabel ("Time (s)", fontsize=18)
    # plt.ylabel ("Sample Value", fontsize=18)
    # plt.title ("Filtered Signal (Interval)", fontsize=22)
    # plt.legend ()
    # plt.show ()
    filt = bwfilter (full)
    r_peaks = Detectors(360).engzee_detector(full[:,1])
    rps = [r for r in r_peaks if r <= 5400 and r >= 3600]
    ys = filt [rps,1]
    plt.plot(filt [3600:5400, 0], filt[3600:5400, 1], label="MLII")
    plt.plot(rps, ys, 'ro')
    plt.xlabel ("# Sample", fontsize=18)
    plt.ylabel ("Sample Value", fontsize=18)
    plt.title ("R-Peaks on a signal interval", fontsize=22)
    plt.legend ()
    plt.show ()

if __name__ == '__main__':
    # show_filter ()
    show_window ()
