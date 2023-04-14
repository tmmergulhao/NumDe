import numpy as np
import matplotlib.pyplot as plt
import sys
from classy import Class

path_NumDe = "/Users/s2223060/Desktop/NumDe"
#Import the NumDe package
sys.path.append(path_NumDe)
import numerical_derivative as nd

#Load the kh array in unit of h/Mpc
kvec = np.loadtxt("/Users/s2223060/Desktop/OneDrive/EFTofLSSandMT_DS/ph006/pk_ph006_00g_X.dat")[:,0]

#Redshift 
z_pk = 0.2

#Define the Abacus Summit fiducial cosmology
cosmological_parameters_Abacus = {
"A_s": 2.083e-09,
"n_s": 0.9649,
"tau_reio": 0.0952,
"omega_b": 0.02237,
"omega_cdm": 0.12,
"h": 0.6736,
"YHe": 0.2425,
"N_ur": 2.0328,
"N_ncdm": 1,
"m_ncdm": 0.06,
"z_pk": z_pk,
"alpha_s":0.0,
"omega_ncdm":0.0006442,
"output": "mPk",
"non linear": "PT",
"IR resummation": "Yes",
"Bias tracers": "Yes",
"cb": "Yes",
"RSD": "Yes",
"AP": "No"
}

counter = 0
def ComputePk(theta):
    global counter
    """
    Computes the linear matter power spectrum as a function of k-vector for a given set of cosmolo-
    gical parameters.

    Parameters
    ----------
    theta : list
        A list of three parameters: As, omega_cdm and h. omega_cdm represents the density of cold dark ma-
        tter in the universe, h represents the Hubble constant in units of 100 km/s/Mpc and As the
        amplitude of primordial fluctuations.

    Returns
    -------
    numpy.ndarray
        The linear matter power spectrum and other operators as a function of k-vector, computed for
         the given cosmological parameters. The array has the same shape as the input vector khvec.

    Notes
    -----
    This function uses the Class() object from the Class code package to compute the linear matter 
    power spectrum as a function of k-vector for a given set of cosmological parameters. The input 
    parameter theta should be a list with two elements: omega_cdm and h. The function returns a 
    NumPy array representing the linear matter power spectrum as a function of k-vector.
    """

    # Extract the input parameters from the input list
    print(theta)
    A, omega_cdm, h = theta[0], theta[1], theta[2]

    #Get the array in units of 1/Mpc
    khvec = kvec*h

    # Set the cosmological parameters in the Class() object
    cosmological_parameters_Abacus["omega_cdm"] = omega_cdm
    cosmological_parameters_Abacus["h"] = h
    cosmological_parameters_Abacus["A_s"] = A*1e-9

    # Create an instance of the Class() object and set its parameters
    M = Class()
    M.set(cosmological_parameters_Abacus)

    # Compute the linear matter power spectrum using one-loop without IR resummation
    M.compute()
    M.initialize_output(khvec, z_pk, len(khvec))
    operators = M.get_pk_mult(khvec, z_pk, len(khvec)) #[Mpc^3]
    return operators

if __name__ == "__main__":
    ND_object = nd.FiniteDerivative()

    #Fiducial cosmogical parameters. I will Taylor expand around them
    A_fid = 2.083
    omegacdm_fid = 0.12
    h_fid = 0.6736

    expansion_table = {"A":[A_fid, A_fid*0.06],
                       "omega_cdm":[omegacdm_fid, omegacdm_fid*0.06],
                       "h":[h_fid, 0.6736*0.06]}

    N_grid = 4
    N_derivative = 2*N_grid
    ND_object.define_setup(N_derivative, N_grid, expansion_table)
    ND_object.parallel = True
    ND_object.processes = 8
    ND_object.evaluate_in_grid(ComputePk, save_file= "output/PK_grid.npz")