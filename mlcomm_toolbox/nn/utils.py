# -*- coding: UTF-8 -*-

import numpy as np


def normalize_data(x):
    """
    Normalizes data. Should not normalize the first row (we assume it is the row of ones).
    x = np.array of size MxN
    Output:
    x_norm     = normalized np.array of size MxN    
    norm_param = distionary with two keys "mean" and "var". Each key contains 
    a np.array of size Mx1 with the mean and variance of each row of data array. 
    For the first row,  set mean=0 and var=1
    """
    ### YOUR CODE HERE ###
    return x, None


def lir_grad(w, y, x): 
    """
    Returs gradient for linear regression with quadratic cost for parameter w and data set y, x.
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    gradT = np array of size Mx1
    """
    ### YOUR CODE HERE ###
    return 0



def gradient_descent(iter_num, l_rate, w_0, gradient_func):
    """
    Performs gradient descent for iter_num iterations with learning rate l_rate from initial
    position w_0. 
    w_0 = np array of size Mx1
    gradient_func(w) is a function which returns gradient for parameter w
    Output:
    w_opt = optimal parameters
    """
    ### YOUR CODE HERE ###
    return 0


def poly_extend_data1D(x, p=2):
    """
    Extend the provided input vector x, wtih subsequent powers of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN such that 1st row = x^0, 2nd row = x^1, ..., (p+1)th row = x^p
    """      
    ### YOUR CODE HERE ###
    # Python does not support pow() of a list, i.e. x =[1,2,...,N]
    # Here x is type numpy.ndarray, i.e. len(x)=1
    x_e = np.vstack([x**i for i in range(p+1)])
    return x_e
    ######################
    # This is a hint: return np.vstack([x**0,x])

def sin_extend_data1D(x, p):
    """
    Extend the provided input vector x, wtih P subsequent sin harmonics of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN
    """      
    ### YOUR CODE HERE ###
    return 0

def lir_cost(w, y, x):
    """
    Computes cost for linear regression with parameters w and data set x,y
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    cost = scalar
    """
    ### YOUR CODE HERE ###
    return 0

def act_fct(x, type_fct):
    """
    Implements different activation functions to be used in Neural Networks. The
    variable x is the function parameter and type_func defines which functions should
    be chosen, i.e., y = f(x). Valid choices are
    
    'identity': y = f(x) = x
    'sigmoid': y = f(x) = 1/(1+exp(-x))
    'tanh': y = f(x) = tanh(x)
    'rect_lin_unit': y = f(x) = max(x,0)


    """
    if type_fct not in ['identity', 'sigmoid', 'tanh', 'rect_lin_unit']:
        raise ValueError('activation function type {} is not known'.format(type_fct))
    
    x = np.asarray(x, dtype=float)

    if type_fct == 'identity':
        y = x
    elif type_fct == 'sigmoid':
        y = 1/(1+np.exp(-x))
    elif type_fct == 'tanh':
        y = np.tanh(x)
    elif type_fct == 'rect_lin_unit':
        y = np.max(np.vstack((x, np.zeros(x.shape))), axis=0)

    return y
    