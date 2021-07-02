from patientclass import Patient
from functools import partial
from collections import Counter
from utils import *
import numpy as np
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

def balance_patient (pn: int, ds: float, n = None):
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

def select_patients(pns, n_outs):
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
        for lab, occ in cnt:
            labs.add (lab)
    return selected_p, labs


def gen_batch (pns, labs, min_samples, ds):
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

    eqfilt = lambda x : lab == x.btype
    # Build batch
    batch = []
    for lab in labs:
        for p in pns:
            for _ in list (filter (eqfilt, p.wins)):
                batch.append (_)

    # Balance batch
    batch_counter = Counter ([w.btype for w in batch]).most_common ()

    # Find reference size for downsampling
    balanced_labset = set()
    refsize=0
    for lab, occ in batch_counter:
        if occ < min_samples:
            break
        refsize = occ
        balanced_labset.add(lab)

    balanced_batch = []
    for lab, occ in batch_counter:
        if lab in balanced_labset:
            # Downsample and add only beat types that meet requirements
            new_length = int (occ - (occ - refsize) * ds)
            for _ in random.sample(list (filter (eqfilt, batch)),new_length):
                balanced_batch.append (_)

    return random.sample (balanced_batch, len (balanced_batch)), balanced_labset

def gen_inputs(batch, labelset):
    labels = [w.btype for w in batch]
    # One hot encoding
    labels = np.asarray(annotations_to_signal(labels, list(labelset)))
    inputs = np.asarray([np.asarray(w.signal) for w in batch])
    # Reshape to fit model
    inputs = inputs.reshape(len(inputs), 114, 1)
    return inputs, labels

def generate_training_batches(patient_list, batch, labelset):
    for patient in patient_list:
        # Lambda to match patient's windows
        patient_filter = lambda x : patient.number == x.patient
        general_batch = [win for win in batch if not patient_filter(win)]
        patient_batch = [win for win in batch if patient_filter(win)]
        general_batch, general_labels = gen_inputs(general_batch, labelset)
        patient_batch, patient_labels = gen_inputs(patient_batch, labelset)
        yield general_batch, general_labels, patient_batch, patient_labels
