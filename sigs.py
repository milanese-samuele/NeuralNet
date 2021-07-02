from dataclasses import dataclass
from scipy import stats

@dataclass()
class Window:
    def __init__(self, sig, teacher, pnr):
        self.signal = stats.zscore(sig [1])
        self.btype = teacher
        self.patient = pnr

    def __repr__(self):
        print("signal: ", self.signal)
        print("beat type:", self.btype)
        print("patient number:", self.patient)
        return ""

    def __eq__ (self, other):
        return self.btype == other.btype
