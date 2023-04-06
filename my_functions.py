import numpy as np
from classy import Class

def function_test(theta, kwargs):
    """
    A test function that computes the product of sine and cosine functions of an input x-array with the two input parameters theta.

    Parameters
    ----------
    theta : list
        A list of two parameters used in the sine and cosine functions.
    kwargs : dict
        A dictionary containing the input x-array.

    Returns
    -------
    numpy.ndarray
        The product of sine and cosine functions with the two input parameters.

    Notes
    -----
    This function is just an example of a test function that can be used to test the numerical differentiation functions.

    Example
    -------
    >>> x_array = np.linspace(0, 1, 10)
    >>> theta = [2, 3]
    >>> kwargs = {"x_array": x_array}
    >>> function_test(theta, kwargs)
    array([ 0.00000000e+00,  1.07088687e-01,  8.69394761e-01,  3.26537279e-01,
           -9.42260940e-01, -4.12118485e-01,  7.47175482e-01,  6.13113707e-01,
           -2.95520207e-01, -8.44190949e-01])
    """

    return np.sin(kwargs["x_array"] * theta[0]) * np.cos(kwargs["x_array"] * theta[1])


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

kvec = np.logspace(-2, np.log10(3), 1000) # array of kvec in h/Mpc
khvec = kvec * cosmological_parameters["h"] # array of kvec in 1/Mpc


def ComputePk(theta):
    """
    Computes the linear matter power spectrum as a function of k-vector for a given set of cosmolo-
    gical parameters.

    Parameters
    ----------
    theta : list
        A list of two parameters: omega_cdm and h. omega_cdm represents the density of cold dark ma-
        tter in the universe, and h represents the Hubble constant in units of 100 km/s/Mpc.

    Returns
    -------
    numpy.ndarray
        The linear matter power spectrum as a function of k-vector, computed for the given cosmolo-
        gical parameters. The array has the same shape as the input vector khvec.

    Notes
    -----
    This function uses the Class() object from the Class code package to compute the linear matter 
    power spectrum as a function of k-vector for a given set of cosmological parameters. The input 
    parameter theta should be a list with two elements: omega_cdm and h. The function returns a 
    NumPy array representing the linear matter power spectrum as a function of k-vector.
    """

    # Extract the input parameters from the input list
    omega_cdm, h = theta[0], theta[1]

    # Set the cosmological parameters in the Class() object
    cosmological_parameters["omega_cdm"] = omega_cdm
    cosmological_parameters["h"] = h

    # Create an instance of the Class() object and set its parameters
    M = Class()
    M.set(cosmological_parameters)

    # Compute the linear matter power spectrum using one-loop without IR resummation
    M.compute()
    M.initialize_output(khvec, 0.02, len(khvec))
    operators = M.get_pk_mult(khvec, 0.02, len(khvec))*h**3

    # Return the linear matter power spectrum as a function of k-vector
    return operators[14]