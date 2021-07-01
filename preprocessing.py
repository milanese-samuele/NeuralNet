from scipy import signal
import numpy as np
import math
from utils import readcsv, extract_annotations
from ecgdetectors import Detectors
from sigs import Window

def prep_data (func) :
    """
    @brief      decorator to use preprocessing functions returning one array with
               data matrix

    @param      function and signal

    @return     the prepped-preprocessed signal
    """
    def wrapper (fsig) :
        fsig [:,1] = func (fsig [:,1])
        fsig [:,2] = func (fsig [:,2])
        return fsig [:]
    return wrapper

@prep_data
def bwfilter (sig) :
    """
    @brief      third-order band-pass butterworth filter with cutoff
                frequencies of 0.4 and 45 Hz

    @param      a signal to filter

    @return     filtered signal
    """
    sos = signal.butter (3, [0.4, 45], 'bp', fs=360, output='sos')
    return signal.sosfilt (sos, sig )


def window_factory(pnumber: int, data_path="./mitbih_database/", **kwargs):
    """
    @brief      extraction and filtering of data from csv file, and segmentation by r-peak algorithm

    @param      pnumber : id number of patient to preprocess
                data_path : path of database
                kwargs {
                sample_rate : sample rate of data
                onset : number of samples to take before r peak
                offset : number of samples to take after r peak}

    @return     returns a matrix of windowed data
    """

    full = readcsv(data_path + str(pnumber) + ".csv")
    annotations = extract_annotations(pnumber)
    r_peaks = Detectors(
        kwargs.get('sample_rate', 360)).engzee_detector(full[:,1])
    return create_windows(r_peaks, bwfilter(full),
                          kwargs.get('onset', 67), #44
                          kwargs.get('offset', 67), #70
                          annotations)


def create_windows(rps, data, onset, offset, anns):
    """
    @brief      builds Window objects with data and annotations
                to be used only by Window object creation
    """
    wins = []
    tmp = [[data[idx-onset:idx+offset,0],
            data[idx-onset:idx+offset,1],
            data[idx-onset:idx+offset,2]] for idx in rps]
    for ts, a in anns:
        for win in tmp:
            if ts in win [0] and len (win [1]) == onset+offset:
                wins.append (Window (win, a))
                break
    return wins
