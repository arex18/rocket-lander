import numpy as np
import nengo
from nengo.dists import Choice


class OSCConfig(object):
    """A class for storing all the specific
    configuration parameters for the neural implementation
    of OSC"""

    # OSC ------------------------------------------------------------------- #

    n_neurons = 2000
    CB = {
        'dimensions': 4,
        'max_rates': np.random.uniform(low=10, high=200, size=n_neurons),
        'n_neurons': n_neurons,
        # 'neuron_type': nengo.Direct(),
        'radius': 5,
        }

    n_neurons = 2500
    M1 = {
        'dimensions': 4,
        'max_rates': np.random.uniform(low=00, high=100, size=n_neurons),
        'n_neurons': n_neurons,
        'neuron_type': nengo.Direct(),
        'radius': .25, 
        }

    n_neurons = 2500
    M1_mult = {
        'encoders': Choice([[1,1],[-1,1],[-1,-1],[1,-1]]),
        'ens_dimensions': 2,
        'n_ensembles': 4,
        'n_neurons': n_neurons,
        'neuron_type': nengo.Direct(),
        'radius': np.sqrt(2),
        }

    # DMPs ------------------------------------------------------------------ #

    oscillator = {
        'dimensions': 2,
        'n_neurons': 500,
        # 'neuron_type': nengo.Direct(),
        'radius': .01,
        }

    forcing_func = {
        'dimensions': 2,
        'n_neurons': 2000,
        }

    y = {
        'dimensions': 1,
        'n_neurons': 1000,
        # 'neuron_type': nengo.Direct(),
        'radius': 5,
        }


    target_ybias = 2.0

    # ----------------------------------------------------------------------- #
    CBmean = np.array([ 0.74859865, 1.83029154, 0.01458004, -0.02742442 ])
    CBscale = np.array([ 0.49086854, 0.65507996, 2.18998496, 3.09952941])

    def CB_scaledown(self, x):
        return (x - self.CBmean) / self.CBscale

    def CB_scaleup(self, x):
        return x * self.CBscale + self.CBmean

    # ----------------------------------------------------------------------- #
    M1mean = np.array([ 0.67332909,  0.52630158,  0.72467918, -0.83424157])
    M1scale = np.array([ 0.36273571,  0.48462947,  0.323381,  0.29300704])

    def M1_scaledown(self, x):
        return (x - self.M1mean) / self.M1scale

    def M1_scaleup(self, x):
        return x * self.M1scale + self.M1mean

    # ----------------------------------------------------------------------- #
    u_scaling = np.array([1., 1.])
    DPmean = np.array([-66.13644345,   5.93401211, -22.82797318, -21.08624692])
    DPscale = np.array([ 29.21381785,  30.42525345,  12.93365678,   9.4640815 ])

    def DP_scaledown(self, x):
        return (x - self.DPmean) / self.DPscale

    def DP_scaleup(self, x, index):
        return self.u_scaling[index%2] * x[0] * \
                (x[1] * self.DPscale[index] + self.DPmean[index])

    def __init__(self, adaptation):

        if adaptation == 'kinematic':
            self.DPmean *= 2.0

        self.DP_scaleup_list = [
                lambda x: self.DP_scaleup(x, 0),
                lambda x: self.DP_scaleup(x, 1),
                lambda x: self.DP_scaleup(x, 2),
                lambda x: self.DP_scaleup(x, 3)]
