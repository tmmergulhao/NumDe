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

MAIN_DIR = os.getcwd()

#Loading the directories 
'''
findiff - ver 1.0
 This code takes the numerical derivative of any function using the finite difference method.

by: Thiago Mergulhao - University of Sao Paulo
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

def try_mkdir(name):
	"""Simple function that tries to create a directory in case there is no other with the same 
	name as given in the input.

	Args:
		name (str): String with the name of the directory to be created
	"""
	this_dir=os.getcwd()+'/'
	this_dir_to_make = this_dir+name
	#print(this_dir_to_make)
	try:
		os.mkdir(this_dir_to_make)
	except:
		pass

def remove_files_in_dir(name):
	"""Delete all files inside a directory

	Args:
		name (str): A string with the name of the directory whose files inside will be deleted.
	"""
	this_dir = os.getcwd()+'/'
	dir_to_remove = this_dir+name
	try:
		for x in os.listdir(dir_to_remove):
			os.remove(dir_to_remove+x)
	except:
		pass


def load_setup(file_name):
	"""Function that load the .json with the specifications needed to perform the Taylor Expansion.

	Args:
		file_name (string): Name of the file containing the specifications for the expansion.

	Returns:
		N_taylor (int), N_grid (int), hash_table (dictionary): Returns the order of the Taylor 
		expansion, The size of the grid that will be used and a hash_table with the step size in the
		expansion and where the expansion is being performed around (point in parameter space). 
	"""
	setup_dir = MAIN_DIR+'/expansion_setup/'
	try:
		with open(setup_dir+file_name+'.json') as json_file:
			hash_table = json.load(json_file,object_pairs_hook=OrderedDict)["expansion_setup"]
		try:
			N_taylor = hash_table.get('N_taylor')
			N_grid = hash_table.get('N_grid')
			hash_table.pop('N_taylor')
			hash_table.pop('N_grid')
		except:
			print('Your expansion setup input must have the keys: N_taylor and N_grid')
			print('Check if the keys are defined properly.')
			sys.exit(-1)
	except:
		pass

	output_dir = os.getcwd()+'/outputs/'+file_name 
	try_mkdir(output_dir)
	
	return N_taylor, N_grid, hash_table

def load_from_grid(setupname,filename):
	"""Load the function computed in a specific point in the parameter grid space.

	Args:
		setupname (str): Setup name used to compute the operators
		filename (list): A list cointaining where in the parameter grid the operators were computed.

	Returns:
		np.array: A numpy array with the function computed in the grid.
	"""
	grid_dir = MAIN_DIR+'/outputs/'+setupname+'/grid/'
	file_dir = grid_dir+filename
	try:
		the_operators = np.load(file_dir+'.npy')
		return the_operators
	except:
		print('Problem reading the operators evaluated at', filename)
		print('Make sure you have this grid point. The files are supposed to be at:')
		print(file_dir)
		sys.exit(-1)

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
	def __init__(self,**kwargs):
		try:
			self.N_taylor, self.N_grid, self.expansion_table = load_setup(kwargs.get("expansion_setup", None))
			self.params, self.Nparams= list(self.expansion_table.keys()), len(self.expansion_table.keys())
			print("Setup loaded from file!")
		except:
			pass

	def define_setup(self, N_taylor, N_grid, expansion_table):
		self.N_taylor, self.N_grid, self.expansion_table = N_taylor, N_grid, expansion_table
		self.params, self.Nparams = list(self.expansion_table.keys()), len(self.expansion_table.keys())

	def create_parameter_grid(self, param):
		"""Create a grid for a specific parameter.

		Args:
			param (list): A list with the reference value and the step size as a percentage of the 
			it.
			N_stencil (int): The size of the stencil.

		Returns:
			array: The 1D grid in parameter space.
		
		Example:
			param_spec = [1, 0.1], N_stencil = 3
			
			output: [0.9, 1, 1.1]
		"""
		N_stencil = 2*self.N_grid + 1
		reference_value, percentage_step_size = self.expansion_table[param]
		step_size = reference_value*percentage_step_size
		this_grid = np.zeros(N_stencil)
		
		for i in range(0,len(this_grid)):
			counter = step_size*(-(N_stencil-1)/2 + i)
			this_grid[i] = reference_value + counter
		return this_grid

	def generate_all_cases(self):
		"""Generate a list of all possibles partial derivatives up to some order in the Taylor expansion
		and for any number of parameters

		Args:
			N (int): The order of the Taylor expansion
			setup (str): The name of the .json file with the specifications for the Taylor expansion.

		Returns:
			list: A list with all combinations of derivatives.
			
			Example:

				For 3 parameters and N = 2:

				output: [[[0], [1], [2]],
						[[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]], 
						[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1], [0, 1, 2], [0, 2, 2], 
						[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]]]
		"""

		output = []
		params_list = range(0,len(self.expansion_table.keys()))
		for i in range(1,self.N_taylor+1):
			output.append([list(this_value) for this_value in \
				itertools.combinations_with_replacement(params_list,i)])
		return output

	def get_normalization(self, derivative):
		"""Compute the normalisation faction for a given derivative

		Args:
			derivative (list): The derivative in list format
			setup (str): The name of the .json file with the specifications for the Taylor expansion.

		Returns:
			float: The normalization of the derivative.
		"""
		step_size = []
		for i in range(0,self.Nparams):
			step_size.append(self.expansion_table[self.params[i]][0]*self.expansion_table[self.params[i]][1])
		normalization = []
		if len(derivative)>1:
			for i in derivative:
				normalization.append(step_size[i])
			return np.prod(normalization)
		else:
			return step_size[int(derivative[0])]

	def evaluate_in_grid(self, function, **kwargs):
		"""Evaluate the function in the grid.

		Args:
			setup (str): The name of the .json file with the specifications for the Taylor expansion.
		"""
		self.GRID = {}

		#Get the total size of the grid
		total_grid_size = int(2*self.N_grid+1)

		#Iterator for all grid points
		index_grid = [list(tup) for tup in itertools.product(range(-int((total_grid_size-1)/2),\
			int((total_grid_size-1)/2) + 1),repeat = self.Nparams)]
		
		#Get the parameter values in the grid
		params_grid = np.zeros((self.Nparams,total_grid_size))
		for index,this_param in enumerate(self.params):
			this_param_grid = self.create_parameter_grid(this_param)
			params_grid[index,:] = this_param_grid

		#Iterate over all positions in the grid
		for x in tqdm(index_grid):
			#Apply a translation to the grid point in order to be consistent with the parameter vectors
			x_translated = [previous + self.N_grid for previous in x]

			func_argument = []
			for index,y in enumerate(x_translated):
				func_argument.append(float(params_grid[index,y]))

			if len(list(kwargs.keys())) != 0:
				self.GRID[str(x)] = function(func_argument, kwargs)
			else:
				self.GRID[str(x)] = function(func_argument)

	def generate_derivative_label(self, derivative):
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
			the_label += 'd'+str(counts[x])+str(self.params[x])
		
		return the_label	
	
	def take_derivative_ALL(self, verbose = False):
		"""Compute all derivatives according to the setup file

		Args:
			setup (str): The name of the .json file with the specifications for the Taylor expansion.
		"""

		#Read the setup dictionary
		stencil_list = []
		
		for i in range(1,self.N_grid+1):
			stencil_list.append(np.arange(-i,i+1,1))

		self.derivatives = {}
		#Iterate over all derivatives
		for this_case in self.generate_all_cases():
			for this_derivative in this_case:
				this_label = self.generate_derivative_label(this_derivative)
				self.derivatives[this_label]= {}
				to_print = 'This case:'+this_label
				if verbose: print(to_print.center(80, '*'))

				#For a given derivative we can compute the normalization
				this_normalization = self.get_normalization(this_derivative)

				for this_stencil_shape in stencil_list:
					stencil_halfsize = int(len(this_stencil_shape)-1)
					if stencil_halfsize < 10:
						stencil_label = '0'+str(stencil_halfsize)
					else:
						stencil_label = str(stencil_halfsize)
					if verbose: print('This stencil has size', len(this_stencil_shape))
					self.derivatives[this_label][stencil_label] = {}
					stencil = finite_difference_coefficients(this_stencil_shape)
					try:
						GRID_coefficients = stencil.get_derivative_coef(this_derivative,self.Nparams)
					except:
						#Ignore the case in which the derivative is impossible to be taken.
						continue

					#Open some random file and copy its shape
					derivative_output = np.zeros_like(self.GRID[list(self.GRID.keys())[0]])
					
					#Compute the derivative using the finite difference method term by term
					for x,coefficients in GRID_coefficients:
						if verbose: print(x,coefficients)
						derivative_output += self.GRID[x]*coefficients/this_normalization

					self.derivatives[this_label][stencil_label]=derivative_output
