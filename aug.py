from patientclass import Patient
from functools import partial
from collections import Counter
import random


def make_batch(pns: list[int], classes: list[int]) :
    """
    @brief      this function creates of batch of windows belonging to certain
                classes from a list of patients

    @param      pns: list of patient numbers
                classes: the classes of windows to extract

    @return     a batch of windows
    """
    pfilter = partial (filter_btype, pns)
    batch = list (map (pfilter, classes))
    for c in batch:
        print (len (c))
    return [win for sub in batch for win in sub]  # Return the flat batch


def filter_btype (pns: list [int], bt: str):
    """
    @brief      returns a list of beats matching the given type

    @param      pns: list of patient numbers
                bt: beat type to match

    @return     a list of matched beats
    """
    matched = []
    for p in pns:
        for win in Patient(p).wins:
            if win.btype == bt:
                matched.append(win)
    return matched

def balance_patient (pn: int, ds: float, n):
    p = Patient (pn)
    # Get n most commmon classes
    cnt = Counter ([w.btype for w in p.wins]).most_common (n)
    new_batch = []
    # for each most common class
    for lab, occ in cnt:
        # get new length
        new_length = int (occ - (occ - cnt [n-1] [1]) * ds)
        eqfilt = lambda x : lab == x.btype
        # downsample
        for _ in random.sample(list (filter (eqfilt, p.wins)),new_length):
            new_batch.append (_)
    return random.sample (new_batch, len (new_batch))
