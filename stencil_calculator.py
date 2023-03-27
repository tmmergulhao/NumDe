import numpy as np
import sys
from math import factorial
from collections import Counter
import itertools

'''
Finite Difference Coefficients
It is a code to compute the Finite Difference Coefficients for mixed and single derivatives
by: Thiago Mergulhao - University of Sao Paulo
'''

class finite_difference_coefficients:
    """This is a method to compute the coefficients needed to compute derivatives using the 
    Finite Difference Method (FDM)
    """
    def __init__(self,stencil):
        """Initialise the method.

        Args:
            stencil (list or np.array): The stencil used to compute the coefficients. The stencil
            matrix is computed using it as input. We use the Central finite difference.

            Example:
                    s2 = [-2, -1, 0, 1, 2] is a second order stencil. Let s2(0) be the zeroth ele-
                    ment, s2(0) = -2, s2(1) be the second element, and so on. The stencil matrix, S
                    is defined in the following way (see attached wikipedia pdf page). Since the 
                    length of s2 is 5, the matrix is 5x5. Its first line, S0, is defined as:
                        S0 = s2^0,
                    
                    its second line, S1, is defined as:
                        S1 = s2^1,
                    where the pattern is repeated until all lines of the matrix are completed.
        """

        #Storage the Stencil and its size
        self.stencil_length = len(stencil)
        self.stencil = stencil

        #Compute the stencil matrix following the procedure described above.
        self.matrix_stencil = np.zeros((self.stencil_length,self.stencil_length))
        for i in range(0,self.stencil_length):
            self.matrix_stencil[i,:] = pow(self.stencil,i)

    def get_derivative_coef(self, derivative_list, nparams, verbose = False):
        """Use the Stencil Matrix computed above to compute and return the associated coefficients 
        used to compute the desired derivative.

        Args:
            derivative_list (list): A list with the desired derivative you want to compute.
            For instance, if you have 3 parameters, a0,a1 and a2, and want to compute:
                
                d/da0da1da2,
            
            then you need to give as input [0,1,2]. The maximum order derivative you can take for a
            given a stencil is fixed by its size. A stencil of size 5 can compute up to the 4th
            derivative. Therefore, for the exemple above, you should at least have a stencil of size
            3, which is the minium: s3 = [-1, 0 1]. With that stencil you can go up to a second
            order deriative for a single parameter: d/da0da0da1da1 or d/da0da0da1da2.

            nparams (int): The number of parameters your function depend on.
            
            verbose (bool, optional): Whether to print the outputs in the terminal or not. 
            Defaults to False.

        Raises:
            ValueError: Raises a problem if you request to compute a derivative that is not compati-
            ble with the storaged Stencil Matrix.

        Returns:
            _type_: _description_
        """        
        #Whether to print the outputs in the terminal or not
        self.verbose = verbose
        self.param_list = list(np.arange(0,nparams,1))
        self.derivative_list = derivative_list

        #Compute how many times a specific parameter appear in the input
        self.derivatives_holder = Counter(self.derivative_list) 
        self.derivatives_holder_keys = list(self.derivatives_holder.keys())
        self.n_taylor = []
        for x in self.derivatives_holder_keys:
            self.n_taylor.append(self.derivatives_holder[x])
        
        if any(x >= self.stencil_length for x in self.n_taylor):
            raise ValueError('This derivative can not be calculated with this stencil!')
        
        #number of derivatives with respect to different parameters
        self.n_distinct_derivatives = len(set(self.derivative_list))
        
        #Dictionary having all the coefficients
        self.coefficients_output = {}

        #Solve the Stencil System for each parameter separetely
        for key in self.derivatives_holder_keys:
            B = np.zeros(self.stencil_length)
            B[self.derivatives_holder[key]] = factorial(self.derivatives_holder[key])
            these_coefficients = np.linalg.solve(self.matrix_stencil,B)
            self.coefficients_output[key] = these_coefficients

        #Rewrite the dictionary in the proper way to be associated with the grid for all parameters
        stencil_half_size = int((self.stencil_length-1)/2)
        the_iterator = itertools.product(range(-stencil_half_size,stencil_half_size+1), \
            repeat = self.n_distinct_derivatives)
        
        #print('Grid points relevant for this calculation:')
        
        #Iterate over the points in grid space that need to be modified to compute the derivative.
        #If you have three parameters, the center of the grid will be [0, 0, 0]. If you take a deri-
        #vative with respect to the midle parameter, only the middle value will be changed when 
        #computing  the derivative. The derivative coefficients are computed for each parameter 
        #separetely and later on are multiplied together.
        self.final_list = []
        for x in the_iterator:  
            filename = np.zeros(nparams,dtype=int)
            this_coef = []
        
            #Include the parameters varying into the total grid
            for index in range(0,self.n_distinct_derivatives):
                filename[self.derivatives_holder_keys[index]] = int(x[index])
                #print(list(filename))
            
            #Get the coefficients by taking the tensorial product of the single-variable cases
            for i,y in enumerate(x):
                to_append = self.coefficients_output[self.derivatives_holder_keys[i]]\
                    [y+stencil_half_size]
                this_coef.append(to_append)

            self.final_list.append([str(list(filename)),np.prod(this_coef)])

        if self.verbose:
            for points,coefs in self.final_list:
                print(points,coefs)

        return self.final_list

if __name__ == '__main__':
    #Initialize the Stencil setup
    test_stencil = np.array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    nparams = 1
    stencil = finite_difference_coefficients(test_stencil)

    derivative_list_1 = [0,0]
    print(stencil.get_derivative_coef(derivative_list_1,nparams))