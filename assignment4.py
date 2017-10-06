import numpy as np
import numpy.linalg as la
import timeit
import unittest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

############################################################
# Problem 1: Gauss-Jordan Elimination
############################################################


    
def gauss_jordan(A):
    ## Add code here ##
    
    
    print ("A shape: ", A.shape)
    A = A.astype(float)
    n = A.shape[0]
    
    #operae elimination elementary matrix 
    for k in range(0, n):
        print ("row: ", A[k, :])
        #get maximum  value for rows i at or below kth column
        iMax = np.argmax(abs(A[k:n, k]))      #indexMax is in subarray now
        print ("indexMax: ", A[k:n, k], iMax, k, A[iMax+k][k])
        if A[iMax+k][k] == 0:
            return None
        
        #swap k and i*
        A[[iMax, k]] = A[[k, iMax]]
        print ("A: ", A)
        
        for j in range(k+1, n):
            f = A[j][k]/A[k][k]
            A[j, :] -= f * A[k, :]
    
    #operae reverse elimination elementary matrix 
    
    '''
    for each row k = n;:::; 1 (i.e. in reverse) do
        Ak = Ak=Akk
        for each row j above k (i.e. j = k − 1;:::; 1) do
            f = A Akk jk
            Aj = Aj − fAk
        end for
        end for

    '''
    
    for k in range(n-1, 0):  
        A[k, :] = A[k, :] / A[k][k]
        for j in range(k-1, 0):
            f = A[j][k]/A[k][k]
            A[j, :] -= f*A[:, k]
    
    return A

    
############################################################
# Problem 2: Ordinary Least Squares Linear Regression
############################################################

def linear_regression_inverse(X,y):
    ## Add code here ##
    return -1
    
def linear_regression_moore_penrose(X,y):
    ## Add code here ##
    return -1
    
def generate_data(n,m):
    """
        Generates a synthetic data matrix X of size n by m 
        and a length n response vector.
    
        Input:
            n - Integer number of data cases.
            m - Integer number of independent variables.
    
        Output:
            X - n by m numpy array containing independent variable
                observasions.
            y - length n numpy array containg dependent variable
                observations.
    """
    X = np.random.randn(n,m)
    beta = np.random.randn(m)
    epsilon = np.random.randn(n)*0.5
    y = np.dot(X,beta) + epsilon
    
    return X,y
    
def time_linear_regression(method,n,m,n_runs):
    """
        Times a linear regression method on synthetic data of size n by m.
        Tests the function n_runs times and takes the minimum runtime.
    
        Usage:
        >>> time_linear_regression('linear_regression_inverse',100,10,100)
    
        Input:
            method  - String specifying the method to be used. Should be 
                      either 'linear_regression_inverse' or
                      'linear_regression_moore_penrose'.
            n       - Integer number of data cases.
            m       - Integer number of independent variables.
            n_runs  - Integer specifying the number of times to test the method.
        
        Ouput:
            run_time - Float specifying the number of seconds taken by the 
                       shortest of the n_runs trials.
    """
    setup_code = "import numpy as np; from __main__ import generate_data, %s; X,y = generate_data(%d,%d)"%(method,n,m)
    test_code = "%s(X,y)"%method
    return timeit.timeit(test_code,number=n_runs,setup=setup_code)     

def problem2_plots():
    ## Add code here ##
    pass

    
if __name__=="__main__":
    
    #
    problem2_plots()