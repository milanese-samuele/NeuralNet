from dataclasses import dataclass
from scipy import stats
import pickle

@dataclass()
class Window:
    def __init__(self, sig, teacher):
        self.signal = stats.zscore(sig [1])
        self.btype = teacher

    def __repr__(self):
        print("signal: ", self.signal)
        print("beat type:", self.btype)
        return ""
