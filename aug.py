from patientclass import Patient
from functools import partial
from collections import Counter
import random


def make_batch(pns: list, classes: list) :
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


def filter_btype (pns: list, bt: str):
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

def gen_tuning_batch (pns, n_outs, min_samples, ds):
    """
    @brief      Generates a balanced batch of windows from patients selected
                through some criteria

    @param      pns : list of patient numbers in the dataset
                n_outs : desired number of output channels
                min_samples : minimal size of beat type occurrence
                ds : downsampling factor

    @return     a balanced batch and the selected patients from which it is
                generated
    """
    labs = set ()
    selected_p = []
    for p in pns:
        patient = Patient (p)
        # Get n most commmon classes
        cnt = Counter ([w.btype for w in patient.wins]).most_common (n_outs)
        # Ignore patients with not enough classes
        if len (cnt) < n_outs:
            continue
        # Add patient beat type labels to set
        selected_p.append (patient)
        for lab, _ in cnt:
            labs.add (lab)

    eqfilt = lambda x : lab == x.btype
    # Build batch
    batch = []
    for lab in labs:
        for p in selected_p:
            for _ in list (filter (eqfilt, p.wins)):
                batch.append (_)

    # Balance batch
    batch_counter = Counter ([w.btype for w in batch]).most_common ()

    # Find reference size for downsampling
    idx = 0
    for _ in range (len (batch_counter)):
        if batch_counter [idx] [1] < min_samples:
            break
        idx += 1

    balanced_batch = []
    for lab, occ in batch_counter:
        # Downsample and add only beat types that meet requirements
        if occ >= min_samples:
            new_length = int (occ - (occ - batch_counter[idx] [1]) * ds)
            for _ in random.sample(list (filter (eqfilt, batch)),new_length):
                balanced_batch.append (_)

    return balanced_batch, selected_p
