import pickle
from sigs import Window
from preprocessing import window_factory

class Patient:
    def __init__ (self, pnr):
        self.number = pnr
        try:
            with open ("./preprocessed_patients/" + str (pnr), 'rb') as infile:
                self.wins = pickle.load (infile).wins
        except:
            print ("Patient not preprocessed yet")
            with open ("./preprocessed_patients/" + str (pnr), 'wb') as outfile:
                self.wins = window_factory(pnr)
                pickle.dump (self, outfile)
