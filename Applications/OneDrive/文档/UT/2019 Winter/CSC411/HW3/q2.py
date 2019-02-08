# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    dist = l2(test_datum, x_train)
    A_j = -dist[0]/(2*tau**2)
    B = np.max(A_j)
    # to use logsumexp
    # exp(Ai*)/sum(exp(Aj*)) = exp{log[exp(Ai*)/sum(exp(Aj*)]}=exp(Ai*-log(sum(exp(Aj*)))
    # Ai* = Ai-B
    Ai = A_j - B
    lseAj = logsumexp(A_j - B)
    a = np.exp(Ai - lseAj)
    A = np.diag(a)
    w = np.linalg.solve(np.transpose(x_train).dot(A).dot(x_train)+lam*np.identity(d), np.transpose(x_train).dot(A).dot(y_train))
    #print("td:")
    #print("w:")
    #print(w.shape)
    y_hat = np.dot(test_datum, w)[0]
    #print("y_hat")
    #print(y_hat)
    return y_hat


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    train_MSL = np.zeros(taus.shape)
    test_MSL = np.zeros(taus.shape)
    # use the first train_id indices of idx to randomly pick data
    valid_size = int(N * val_frac)
    train_size = N - valid_size
    train_X = np.zeros((train_size, d))
    train_y = np.zeros((train_size, 1))
    valid_X = np.zeros((valid_size, d))
    valid_y = np.zeros((valid_size, 1))
    for i in range(N):
        if i < train_size:
            train_X[i, ] = x[idx[i], ]
            train_y[i] = y[idx[i]]
        else:
            valid_X[i-train_size,] = x[idx[i], ]
            valid_y[i-train_size] = y[idx[i]]
    # Split finished

    for i in range(len(taus)):
        train_losses = np.zeros(train_size)
        test_losses = np.zeros(valid_size)
        # Train error
        for j in range(train_size):
            yhat = LRLS(np.array([train_X[j, :]]), train_X, train_y, taus[i])
            loss = ((yhat - train_y[j, 0])**2)
            train_losses[j] = loss
        # Valid Error
        for j in range(valid_size):
            #print("vX:")
            yhat = LRLS(np.array([valid_X[j, :]]), train_X, train_y, taus[i])
            loss = ((yhat - valid_y[j, 0])**2)
            test_losses[j] = loss
            #print(y_hat_j)
        train_MSL[i] = np.mean(train_losses)
        test_MSL[i] = np.mean(test_losses)
    return train_MSL, test_MSL



if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
#    dist = l2(x[1,:].reshape(1,d),x)

    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses)
    plt.xlabel('log(Tau)')
    plt.ylabel('Train Mean Square Losses')
    plt.show()
    plt.semilogx(taus, test_losses)
    plt.xlabel('log(Tau)')
    plt.ylabel('Valid Mean Square Losses')
    plt.show()
