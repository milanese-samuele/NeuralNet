from scipy import signal
import numpy as np
import math
from utils import readcsv, extract_annotations
from ecgdetectors import Detectors

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


def preprocess(pnumber: int, data_path="./mitbih_database/", **kwargs):
    """
    @brief      extraction and filtering of data from csv file, and segmentation by r-peak algorithm

    @param      pnumber : id number of patient to preprocess
                data_path : path of database
                kwargs {
                sample_rate : sample rate of data
                onset : number of samples to take before r peak
                offset : number of samples to take after r peak
                }

    @return     returns a matrix of windowed data
    """
    full = readcsv(data_path + str(pnumber) + ".csv")
    r_peaks = Detectors(kwargs.get('sample_rate', 360)).engzee_detector(full[:,1])
    wins = create_windows(r_peaks, bwfilter(full), kwargs.get('onset', 44), kwargs.get('offset', 70))
    return wins


def create_windows(rps: list[int], data: list, onset: int, offset: int):
    ret = []
    for idx in rps:
        entry = [data[idx-onset:idx+offset,0], data[idx-onset:idx+offset,1]]
        ret.append(entry)
    return np.asarray(ret)
