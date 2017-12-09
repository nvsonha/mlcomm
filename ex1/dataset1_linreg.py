# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)
np.set_printoptions(precision=5, linewidth=160, suppress=True)

class DataSet:
    '''

    _x_D, _y_D: sinusoidal data with noise
    x_GT, y_GT: ground truth (i.e. no-noise) sinusoidal data
    '''
    __x_range = [0, 1]  # Array of 2 values 0 and 1
    __sigma = 0.1       # Standard deviation, support Gaussian distr.
    __N_samples = 10    # 10 points
    _y_D = None         # Initialized empty x
    _x_D = None         # Initialized empty y    
    
    @staticmethod
    def __gt_func(x):
        '''

        Extract sinusoidal y based on the given x
        '''
        return np.sin(2*np.pi*x)    # Return sinusoidal y

    @classmethod
    def get_data(cls):  # Parameter cls = self
        '''

        Uniformly randomly generate __N_samples, i.e. 10, _x_D points
        Get sinusoidal of _x_D
        Add Gaussian noise to obtain _y_D
        '''
        if cls._y_D is None and cls._x_D is None:                                                   # Data x, y is empty
            cls._x_D = np.random.uniform(cls.__x_range[0], cls.__x_range[1], (1,cls.__N_samples))   # Uniform 10 _x_D points
            cls._y_D = cls.__gt_func(cls._x_D) + np.random.normal(0, cls.__sigma, cls._x_D.shape)   # A normal (Gaussian) distribution
        return cls._y_D,cls._x_D                                                                    # Return _y_D and _x_D
    
    @classmethod
    def get_ground_truth_data(cls):

        '''

        No-noise data generation
        '''
        x = np.linspace(cls.__x_range[0], cls.__x_range[1]*1.0, 1000).reshape((1,-1))   # Without reshape, len(x)=1000; with reshape, len(x)=1
        y = cls.__gt_func(x)                                                            # Regards to size: len(y)=1
        return y,x                                                                      # Both len(x)=1 and len(y)=1

    @classmethod
    def plot_data(cls):
        y,x = cls.get_data()                                # Data with noise
        y_GT,x_GT = cls.get_ground_truth_data()             # Ground_truth Data (actually no-noise data must be '_y_D minus noise')
        plt.plot(x[0], y[0], 'o', x_GT[0], y_GT[0],'r--')   # Noise data is represented by 10 points; Ground_truth Data by red line
        plt.legend(['data','ground_truth (unknown)'])       # Label 
                
    @classmethod
    def plot_model(cls, w, extend_data, norm_param=None):  
        w = np.array(w).reshape((-1,1))  
        cls.plot_data()
        x = np.linspace(cls.__x_range[0],cls.__x_range[1], 100).reshape((1,-1))
        xe = extend_data(x)
        if norm_param is not None:
            mean = norm_param['mean'].reshape((-1,1))
            stdd = np.sqrt(norm_param['var'].reshape((-1,1)))
            xe = (xe - mean)/stdd
        y = np.dot(w.T, xe)
        plt.plot(x[0], y[0], '-')
        plt.legend(['data','ground_truth (unknown)', 'model'])