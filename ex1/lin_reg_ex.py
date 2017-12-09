# -*- coding: utf-8 -*-
import sys
sys.path.append('../')                  # Add parent folder path  
import numpy as np
import matplotlib.pyplot as plt
import mlcomm_toolbox.nn.utils as nnu   # Folder mlcomm_toolbox is under the parent path
from dataset1_linreg import DataSet  

#get and plot the data
y_D,x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()


#extend x with ones:
x_D = nnu.poly_extend_data1D(x_D)


#random init of w:
### YOUR CODE HERE ###

#normalization:
x_D, norm_param = nnu.normalize_data(x_D)


#plot and compute cost
def extension_wrapper(x):
    return nnu.poly_extend_data1D(x)
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))


#compute gradient and do gradient descent
def gradient_wrapper(w):
    return nnu.lir_grad(w, y_D, x_D)
w = nnu.gradient_descent(1000, 0.05, w, gradient_wrapper)


#plot and compute cost
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))
