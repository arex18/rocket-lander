import numpy as np
import nengo
from nengo.dists import Choice


class OSCConfig(object):
    """A class for storing all the specific
    configuration parameters for the neural implementation
    of OSC"""

    # OSC ------------------------------------------------------------------- #

    n_neurons = 3000
    # n_neurons = 5000
    CB = {
        'dimensions': 6,
        'max_rates': np.random.uniform(low=10, high=200, size=n_neurons),
        'n_neurons': n_neurons,
        # 'neuron_type': nengo.Direct(),
        'radius': 5,
        }

    # n_neurons = 2000
    M1 = {
        'dimensions': 6,
        'max_rates': np.random.uniform(low=00, high=100, size=n_neurons),
        'n_neurons': n_neurons,
        # 'neuron_type': nengo.Direct(),
        'radius': .25, 
        }

    # n_neurons = 7000
    M1_mult = {
        'encoders': Choice([[1,1],[-1,1],[-1,-1],[1,-1]]),
        'ens_dimensions': 2,
        'n_ensembles': 6,
        'n_neurons': n_neurons,
        # 'neuron_type': nengo.Direct(),
        'radius': np.sqrt(2),
        }

    # n_neurons = 3000
    M1_null = {
        'dimensions': 3,
        'max_rates': np.random.uniform(low=10, high=200, size=n_neurons),
        'n_neurons': n_neurons,
        # 'neuron_type': nengo.Direct(),
        'radius': np.sqrt(3),
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
    CBmean = np.array([.61, 1.86, .416, -0.03, -.012, .021]) 
    CBscale = np.array([.4, .6, .3, 1.0, 1.0, 1.0])

    def CB_scaledown(self, x):
        return (x - self.CBmean) / self.CBscale

    def CB_scaleup(self, x):
        return x * self.CBscale + self.CBmean

    # ----------------------------------------------------------------------- #
    M1mean = np.array([.58, .57, .23, .79, -.76, -.96]) 
    M1scale = np.array([.43, .525, .3, .25, .4, .08])

    def M1_scaledown(self, x):
        return (x - self.M1mean) / self.M1scale

    def M1_scaleup(self, x):
        return x * self.M1scale + self.M1mean

    # ----------------------------------------------------------------------- #
    M1nullmean = np.array([0.61,  1.9,  0.4]) 
    M1nullscale = np.array([.45, .55, .32])

    def M1null_scaledown(self, x):
        return (x - self.M1nullmean) / self.M1nullscale

    def M1null_scaleup(self, x):
        return x * self.M1nullscale + self.M1nullmean

    # ----------------------------------------------------------------------- #
    u_scaling = np.array([2., 1.])
    DPmean = np.array([-167., -26., -90., -81., -20., -33.])
    DPscale = np.array([50., 65., 35., 35., 5., 9.]) 

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
                lambda x: self.DP_scaleup(x, 3),
                lambda x: self.DP_scaleup(x, 4),
                lambda x: self.DP_scaleup(x, 5)]
