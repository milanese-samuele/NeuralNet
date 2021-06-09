import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

## Globals for parameter tuning
HD = 10 # Hidden layer dimension
ID = 2   # Input layer dimension
OD = 1   # Output layer dimension
SPARS = 0.2

class RNN:
    """
    @brief      Architecture of the Recurrent Neural Network
    """
    Whid = sparse.random (HD, HD, density=SPARS)   # weight matrix of hidden layer
    Win = np.random.uniform (0, 1, (HD, ID))   # weight matrix for input to hidden layer
    Wout = np.random.uniform (0, 1, (OD, HD))  # weight matrix for hidden layer to output level
    xvec = np.random.uniform (0, 1, HD)        # vector of activation values of neurons
    bvec = np.asarray (HD * [0])               # bias vector

    def sigmoid (vec : np.ndarray) -> np.ndarray :
       """
       @brief      sigmoid function to compute next state of activation vector

       @param      a vector of activations

       @return     a vector of activations after sigmoid function has been applied
       """
       op = lambda x :  1.0 / (1.0 + np.exp (-x))
       return list (map (op, vec))

    def __repr__ (self) :
        """
        @brief      magic method for represention

        @param      self

        @return     placeholder
        """
        print (self.Win)
        print (self.Whid)
        print (self.Wout)
        plt.plot (range (len (self.xvec)), self.xvec [:])
        plt.show ()
        return ""

    def update_state (self, ivec : np.ndarray) :
        """
        @brief      Update state formula

        @param      self
        """
        self.xvec = RNN.sigmoid (self.Whid.dot (self.xvec) + self.Win.dot (ivec) + self.bvec)

    def output (self) -> np.ndarray :
        """
        @brief      compute output function

        @return     output signal
        """
        return RNN.sigmoid (self.Wout.dot (self.xvec))

    def test_run (self, isig : np.ndarray) :
        osig = []
        for tup in isig [:720] :
            self.update_state (tup)
            osig.append (self.output ())

        plt.plot (range (len (osig)), osig )
        plt.show ()
        # print (self)
