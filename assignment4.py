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
    
    
    #print ("A shape: ", A.shape)
    A = A.astype(float)
    n = A.shape[0]
    
    
    #augment identity matrix
    identityMat = np.identity(n).astype(float)
    #print ("identityMat: ", identityMat)
    AMat = np.concatenate((A, identityMat), axis=1)
    
    #operae elimination elementary matrix 
    for k in range(0, n):
        #print ("row: ", AMat[k, :])
        #get maximum  value for rows i at or below kth column
        iMax = np.argmax(abs(AMat[k:n, k]))      #indexMax is in subarray now
        #print ("indexMax: ", AMat[k:n, k], iMax, k, AMat[iMax+k][k])
        if AMat[iMax+k][k] == 0:
            return None
        
        #swap k and i*
        AMat[[iMax+k, k]] = AMat[[k, iMax+k]]
        #print ("AMat: ", AMat)
        
        for j in range(k+1, n):
            f = AMat[j][k]/AMat[k][k]
            AMat[j, :] -= f * AMat[k, :]
    
        #operae reverse elimination elementary matrix 
    
    #print ("AMat after: ", AMat)
    for k in range(n-1, -1, -1):  
        AMat[k, :] = AMat[k, :] / AMat[k,k]
        #print ("AMat later: ", k,  AMat)
        for j in range(k-1, -1, -1):
            f = AMat[j, k]/AMat[k, k]
            AMat[j, :] -= f*AMat[k, :]
    print ("AMat final: ", k,  AMat)
    return AMat[:, n:]

    
############################################################
# Problem 2: Ordinary Least Squares Linear Regression
############################################################

def linear_regression_inverse(X,y):
    ## Add code here ##
    
    inter =  np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
    return np.dot(inter, y)

    
def linear_regression_moore_penrose(X,y):
    ## Add code here ##
    inter = np.linalg.pinv(X)    
    return np.dot(inter, y)
    
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
    
    
    n = 1000
    timeInverseY = []
    timeMooreY = []
    timeX = range(25, 251)
    for m in timeX:
        timeInverseY.append(time_linear_regression('linear_regression_inverse', n, m, 10))
        timeMooreY.append(time_linear_regression('linear_regression_moore_penrose', n, m, 10))
        
    plt.figure()
    plt.plot(timeX, timeInverseY, marker='o', linestyle='--', color='r', label='Inverse')
    plt.plot(timeX, timeMooreY, marker='x', linestyle='--', color='g', label='Moore_penrose')
    plt.legend(loc='upper right')
    plt.xlabel("Covariate number m")
    plt.ylabel("Time (s)")
    plt. title ("Time for fixed n and varing m")
    plt.show()
    
    

    m = 25
    timeInverseY = []
    timeMooreY = []
    timeX = range(1000, 10001)
    for n in timeX:
        timeInverseY.append(time_linear_regression('linear_regression_inverse', n, m, 10))
        timeMooreY.append(time_linear_regression('linear_regression_moore_penrose', n, m, 10))
        
    plt.figure()
    plt.plot(timeX, timeInverseY, marker='o', linestyle='--', color='r', label='Inverse')
    plt.plot(timeX, timeMooreY, marker='x', linestyle='--', color='g', label='Moore_penrose')
    plt.legend(loc='upper right')
    plt.xlabel("Number of data n")
    plt.ylabel("Time (s)")
    plt. title ("Time for fixed m and varing n")
    plt.show()


if __name__=="__main__":
    
    #
    problem2_plots()
