from dataclasses import dataclass
import pickle

@dataclass()
class Window:
    def __init__(self, sig, teacher):
        self.signal = sig
        self.btype = teacher

    def __repr__(self):
        print("signal: ", self.signal)
        print("beat type:", self.btype)
        return ""
