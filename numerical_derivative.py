import sys
from stencil_calculator import finite_difference_coefficients
import numpy as np
import itertools, os, json
from collections import OrderedDict, Counter
from tqdm import tqdm
from math import factorial
from scipy.special import binom
from time import perf_counter
import matplotlib.pyplot as plt
import h5py as h5
from multiprocessing import Pool
from typing import List, Dict, Any

MAIN_DIR = os.getcwd()

#Loading the directories 
'''
NumDe - ver 1.0
 This code takes the numerical derivative of any function using the finite difference method.

by: Thiago Mergulhao - University of Sao Paulo, University of Edinburgh
'''

####################################################################################################
#Auxiliary functions			          
####################################################################################################
def multinomial(params):
	"""Function to compute the multinomial coeficients used in the Taylor expansion of multiple
	parameters.

	Args:
		params (list): List of index containing the powers of each parameter: [n1, n2, ..., n_n].
		For example, if you have three parameters with powers n1, n2 and n3, the associated multino-
		mial will be computed using [n1, n2, n3] as input.

	Returns:
		_type_: Return (n1, n2, ..., n_n)! = Comb(n1+...+n_n; n1, n2, ..., n_n)
	"""
	if len(params) == 1:
		return 1
	return binom(sum(params), params[-1]) * multinomial(params[:-1])

def generate_derivative_label(derivative, params):
	"""Generate a label in the form of string to name the outputs.

	Args:
		derivative (list): A list with the derivatives.

		setup (string): File name of the specs file

	Returns:
		string: A string that is used as a label to tag the derivative outputs.
	
	Example: 
	For a given setup with param0 and param1:
		* [0,0] gives the label for the second derivative of parameter 0: dparam0dparam0
		* [0,1] gives dparam0dparam1, etc...
	"""

	counts = Counter(derivative)
	the_label = ''
	for x in counts.keys():
		the_label += 'd'+str(counts[x])+str(params[x])
	
	return the_label

class FiniteDerivative:
	def __init__(self, setup = None):
		try:
			self.N_derivative, self.N_grid, self.expansion_table = setup
			self.params, self.Nparams= list(self.expansion_table.keys()), len(self.expansion_table.keys())
			print("Setup loaded from file!")
		except:
			pass

	def define_setup(self, N_derivative, N_grid, expansion_table):
		self.N_derivative, self.N_grid, self.expansion_table = N_derivative, N_grid, expansion_table
		self.params, self.Nparams = list(self.expansion_table.keys()), len(self.expansion_table.keys())
		self.parallel = False
		self.processes = 8

	def create_parameter_grid(self, param):
		"""Create a grid for a specific parameter.

		Args:
			param (list): A list with the reference value and the step size
			N_stencil (int): The size of the stencil.

		Returns:
			array: The 1D grid in parameter space.
		
		Example:
			param_spec = [1, 0.1], N_stencil = 3
			
			output: [0.9, 1, 1.1]
		"""
		N_stencil = 2*self.N_grid + 1
		reference_value, step_size = self.expansion_table[param]
		this_grid = np.zeros(N_stencil)
		
		for i in range(0,len(this_grid)):
			counter = step_size*(-(N_stencil-1)/2 + i)
			this_grid[i] = reference_value + counter
		return this_grid

	def generate_all_cases(self, N = None):
		"""Generate a list of all possible partial derivatives up to a specified order in the Taylor expansion for any number
		of parameters.

		Args:
			None

		Returns:
			list: A nested list containing all combinations of partial derivatives.
			
			Example:
			
			For 3 parameters and N = 2:
			
			output: [[[0], [1], [2]],
					[[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]], 
					[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1], [0, 1, 2], [0, 2, 2], 
					[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]]]

		This function generates all possible partial derivatives up to the specified order of the Taylor expansion. The function
		returns a nested list containing all combinations of partial derivatives. The order of the Taylor expansion is set by
		the variable N_derivative, which is an attribute of the object. The number of parameters is determined by the size of the
		expansion_table dictionary, which is also an attribute of the object. Each element of the nested list represents a partial
		derivative, and each sublist contains the indices of the parameters that are involved in that derivative. For example, 
		the sublist [0, 1] represents the partial derivative with respect to the first and second parameters.
		"""

		output = []
		params_list = range(0,len(self.expansion_table.keys()))
		if N is not None:
			for i in range(1,N+1):
				output.append([list(this_value) for this_value in \
					itertools.combinations_with_replacement(params_list,i)])

		else:
			for i in range(1,self.N_derivative+1):
					output.append([list(this_value) for this_value in \
						itertools.combinations_with_replacement(params_list,i)])
		return output

	def get_normalization(self, derivative: List[int]) -> float:
		"""Compute the normalization factor for a given derivative.

		Args:
			derivative (List[int]): The derivative in list format.

		Returns:
			float: The normalization factor of the derivative.
		"""
		step_sizes = [self.expansion_table[self.params[i]][1] for i in range(self.Nparams)]
		normalization = np.prod([step_sizes[i] for i in derivative])
		return normalization

	def save_grid_to_file(self, filename):
		"""
		Save the grid data to a file in .npz format.

		Args:
			filename (str): The name of the file to save the grid data to.

		Raises:
			ValueError: If no grid has been computed yet.

		Returns:
			None
		"""
		if not self.GRID:
			raise ValueError("No grid has been computed yet.")

		# Create a dictionary with the data to be saved
		data = {"params": self.params,
				"GRID": self.GRID,
				"N_derivative":self.N_derivative,
				"N_grid":self.N_grid,
				"expansion_table":self.expansion_table}

		# Save the data to the file
		with open(filename, "wb") as f:
			np.savez(f, **data)

	def load_grid_from_file(self, filename):
		"""Load the grid of evaluation results from a file.

		Args:
			filename (str): The name of the file to load.

		Returns:
			dict: A dictionary containing the evaluation results for each grid point.
		"""
		try:
			with np.load(filename, allow_pickle = True) as f:
				if 'GRID' not in f:
					raise ValueError("File does not contain a valid GRID")
				self.GRID = f["GRID"].item()
				self.params = f["params"]
				self.N_derivative = f["N_derivative"].item()
				self.N_grid = f["N_grid"].item()
				self.expansion_table = f["expansion_table"].item()
				self.Nparams = len(self.params)
				if "derivatives" in f.files:
					print("Derivatives loaded!")
					self.derivatives = f["derivatives"].item()

		except (IOError, ValueError, KeyError) as e:
			print(f"An error occurred while loading the file {filename}: {e}")
			return None

	def evaluate_in_grid(self, function, save_file = None, **kwargs):
		"""Evaluate the given function on a grid of parameter values using caching and multiproces-
		ing to speed up the evaluation process.

		Args:
			function (callable): The function to evaluate. It must take an array of parameter values
			as its first argument.
			
			**kwargs: Additional keyword arguments to pass to the function.

		Attributes:
			self.Nparams (int): The number of parameters that the function takes as input.
			self.params (list): A list of tuples, where each tuple contains the name and domain of a parameter.
			self.N_grid (int): The number of grid points in each dimension.
			self.parallel (bool): A boolean flag indicating whether to use multiprocessing to speed up the evaluation process.
			self.processes (int): The number of processes to use if self.parallel is True.
			self.GRID (dict): A dictionary containing the evaluation results for each grid point.

		Returns:
			None
		"""
		#Initialize the grid
		self.GRID = {}

		# Get the total size of the grid
		total_grid_size = int(2*self.N_grid + 1)

		# Iterator for all grid points
		index_grid = [list(tup) for tup in itertools.product(range(-int((total_grid_size-1)/2),\
			int((total_grid_size-1)/2) + 1), repeat=self.Nparams)]

		# Get the parameter values in the grid
		params_grid = np.zeros((self.Nparams, total_grid_size))
		for index, this_param in enumerate(self.params):
			this_param_grid = self.create_parameter_grid(this_param)
			params_grid[index, :] = this_param_grid

		# Vectorize the function evaluation
		func_arguments = np.zeros((len(index_grid), self.Nparams))
		for i, x in enumerate(index_grid):
			x_translated = [previous + self.N_grid for previous in x]
			for j, y in enumerate(x_translated):
				func_arguments[i, j] = float(params_grid[j, y])

		# Use caching to avoid re-evaluating the same parameter values
		cache = {}
		results = []
		new_results = []
		if self.parallel:
			with Pool(self.processes) as p:
				# create the input arguments for the function
				chunk_args = np.zeros((len(index_grid), self.Nparams))
				for i, x in enumerate(index_grid):
					x_translated = [previous + self.N_grid for previous in x]
					for j, y in enumerate(x_translated):
						chunk_args[i, j] = float(params_grid[j, y])
				if len(kwargs) > 0:
					results = list(tqdm(p.starmap(function, zip(chunk_args, itertools.repeat(kwargs)),
											chunksize=max(1, len(index_grid) // self.processes)), total=len(index_grid)))
				else:
					results = list(tqdm(p.map(function, chunk_args,
										chunksize=max(1, len(index_grid) // self.processes)), total=len(index_grid)))
		else:
			# Compute the function in the grid without multiprocessing
			for i, arg in enumerate(tqdm(func_arguments)):
				arg_str = str(arg)
				if arg_str in cache:
					new_results.append(cache[arg_str])
				else:
					if len(kwargs) > 0:
						result = function(arg, kwargs)
					else:
						result = function(arg)
					new_results.append(result)
					cache[arg_str] = result
				results.append(result)

		# Store the results in a dictionary
		for i, x in enumerate(index_grid):
			self.GRID[str(x)] = results[i] if self.parallel or str(func_arguments[i]) not in cache else cache[str(func_arguments[i])]
	
		if save_file is not None:
			self.save_grid_to_file(save_file)

	def generate_derivative_label(self, derivative: List[int]) -> str:
		"""Generates a label to name the derivative outputs.

		Args:
			derivative (list): A list of integers indicating the derivatives to be taken.

		Returns:
			str: A string that is used as a label to tag the derivative outputs.
			The label format follows the convention of naming each derivative 
			as 'd[n]param[i]' where [n] is the number of times the derivative is taken,
			and [i] is the index of the parameter with respect to which the derivative is taken.
			For example, [0, 0] gives the label for the second derivative of parameter 0: 'd2param0'.
		"""
		counts = Counter(derivative)
		label = ''.join(['d' + str(counts[x]) + str(self.params[x]) for x in counts.keys()])
		return label

	def PlotDerivatives(self, x_axis = None):
		for this_derivative in list(self.derivatives.keys()):
			plt.figure(figsize = (4,4), constrained_layout = True)
			plt.title(this_derivative)
			for this_stencil in list(self.derivatives[this_derivative]):
				try:
					if x_axis is not None:
						plt.plot(x_axis, self.derivatives[this_derivative][this_stencil], \
							label = this_stencil)
					else:
						plt.plot(self.derivatives[this_derivative][this_stencil], \
							label = this_stencil)
				except:
					pass
			plt.legend()
			plt.show()
		plt.close('all')
	
	def take_derivative_ALL(self, verbose: bool = False, save_file = None) -> Dict[str, Dict[str, Any]]:
		"""
		Compute all derivatives according to the setup file.

		Args:
			verbose: A boolean flag indicating whether to print verbose output.

		Returns:
			A dictionary containing the computed derivatives.
		"""

		stencil_list = [np.arange(-i, i+1, 1) for i in range(1, self.N_grid+1)]
		all_cases = self.generate_all_cases()
		self.derivatives = {}

		#A cache for vanishing derivatives. All subsequent derivatives of this case will vanish 
		#automatically
		self.vanishing_derivatives = []

		reference_label = str(list(np.zeros(self.Nparams,dtype=int)))
		reference_point = self.GRID[reference_label]
		
		for this_case in all_cases:
			for this_derivative in this_case:
				this_label = self.generate_derivative_label(this_derivative)
				self.derivatives[this_label] = {}
				if verbose:
					print(f"This case: {this_label}".center(80, "*"))

				this_normalization = self.get_normalization(this_derivative)

				for this_stencil_shape in stencil_list:
					stencil_halfsize = len(this_stencil_shape) - 1
					stencil_label = f"0{stencil_halfsize}" if stencil_halfsize < 10 else str(stencil_halfsize)
					if verbose:
						print(f"This stencil has size {len(this_stencil_shape)}")
					self.derivatives[this_label][stencil_label] = {}
					stencil = finite_difference_coefficients(this_stencil_shape)

					try:
						GRID_coefficients = stencil.get_derivative_coef(this_derivative, self.Nparams)
					except ValueError:
						# Ignore the case in which the derivative is impossible to be taken.
						continue

					derivative_output = np.zeros_like(self.GRID[list(self.GRID.keys())[0]])
					for x, coefficients in GRID_coefficients:
						if verbose:
							print(x, coefficients)
						derivative_output += self.GRID[x] * coefficients / this_normalization

					derivative_output = np.around(derivative_output,5)
					if np.all(derivative_output == 0) and this_derivative not in self.vanishing_derivatives:
						self.vanishing_derivatives.append(this_derivative)

					self.derivatives[this_label][stencil_label] = np.around(derivative_output,5)
		
		if save_file is not None:
			f = np.load(save_file)
			data = dict(np.load(save_file,allow_pickle = True).items())
			
			data.update({"derivatives":self.derivatives})
			np.savez(save_file, **data)
			f.close()
	
	def TaylorExpand(self, x, N_taylor, stencil = None):

		#Which stencil size use when loading the derivatives. The standard value is the largest 
		#one, stencil = 2*N_grid.
		if stencil is None:
			stencil = str(2*self.N_grid).zfill(2)
		else:
			stencil = str(stencil).zfill(2)
			print(stencil)
		#Load the function computed at the reference point
		reference_label = str(list(np.zeros(self.Nparams,dtype=int)))
		reference_point = self.GRID[reference_label]

		#Accumulate all contributions of the Taylor expansion
		taylor_corrections = 0

		#Calculate the difference between the input and the reference value
		diffs = []
		for i in range(len(x)):
			diffs.append(x[i] - self.expansion_table[self.params[i]][0])
	
		#Iterate over all terms contributing to the Taylor expansion up to some order
		for index, derivative_list in enumerate(self.generate_all_cases(N_taylor)):
			derivative_order = index+1
			denominator_factor = factorial(derivative_order)

			for derivative in derivative_list:

				#Generate the derivative label
				derivative_label = generate_derivative_label(derivative, self.params)

				#List to storage the polynomial terms and the associated multinomial term				
				polynomial_part = []
				multinomial_index = []

				#Count the power for the polynomial part
				pows = Counter(derivative)

				#Compute the polynomial part
				for expanded_variable in pows.keys():
					polynomial_part.append(pow(diffs[expanded_variable],pows[expanded_variable]))
					multinomial_index.append(pows[expanded_variable])

				taylor_corrections += np.prod(polynomial_part)*multinomial(multinomial_index)\
					*self.derivatives[derivative_label][stencil]/denominator_factor

		return reference_point + taylor_corrections