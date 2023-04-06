import numpy as np
from classy import Class

def function_test(theta,kwargs):
    return np.sin(kwargs["x_array"]*theta[0]) * np.cos(kwargs["x_array"]*theta[1])


cosmological_parameters = {"A_s": 2.6085e-09,
                            "n_s": 0.9611,
                            "tau_reio": 0.0952,
                            "omega_b": 0.02216104779708,
                            "omega_cdm": 0.11888851928939997,
                            "h": 0.6777,
                            "YHe": 0.2425,
                            "N_ur": 2.0328,
                            "N_ncdm": 1,
                            "m_ncdm": 0.06,
                            "z_pk": 0.02,
                            "output": "mPk",
                            "non linear": "PT",
                            "IR resummation": "Yes",
                            "Bias tracers": "Yes",
                            "cb": "No",
                            "RSD": "Yes",
                            "AP": "No"}

kvec = np.logspace(-2,np.log10(3),1000) # array of kvec in h/Mpc
khvec = kvec*cosmological_parameters["h"] # array of kvec in 1/Mpc

#We first define a function that updates omega_cdm and return the new P(k)
def ComputePk(theta):
    omega_cdm, h = theta[0], theta[1]
    cosmological_parameters["omega_cdm"] = omega_cdm
    cosmological_parameters["h"] = h
    M = Class()
    M.set(cosmological_parameters)
    #let's first take a look at the one-loop power spectrum for matter without IR resummation
    M.compute()
    M.initialize_output(khvec, 0.02, len(khvec))
    operators = M.get_pk_mult(khvec, 0.02, len(khvec))*h**3
    return operators[14]