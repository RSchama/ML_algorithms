#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:04:31 2022

@author: schama
"""

#Code for the Logistic Regression notebook#
#libraries needed and their aliases
import numpy as np
import numpy.matlib
import math
from sklearn.metrics import accuracy_score

#Likelihood function
def likelihood(wt,x,y):
    '''
    Calculates the log-likelihood of the parameter vector wt for the
    logistic regression with gradient ascent.

    Parameters
    ----------
    wt : float
        vector of parameters for the logistic regression. From beta_0 to beta_p, at time t.
    x : numeric
        Data matrix with N observations and p features, used for training the model. 
        A column of ones at index 0 is expected.
    y : binary 0/1
        Label vector of x, used for training the logistic regression classification model.

    Returns
    -------
    -1 * log-likelihood of the wt parameter vector.

    '''
    L=0
    N, p = x.shape
    for j in range(N):
        L+=y[j]*np.dot(x[j,],wt) - math.log(1 + math.exp(np.dot(x[j,],wt)))
    #print(np.dot(x[j,],wt))
    return(-1*L)

#derivative function
def derivative_func(wt,x,y):
    '''
    Calculates the derivative needed for the update_w() function

    Parameters
    ----------
    wt : float
        Vector of parameters for the logistic regression. From beta_0 to beta_p, at time t.
    x : numeric
        Data matrix with N observations and p features, used for training the model.
        A column of ones at index 0 is expected.
    y : binary 0/1
        Label vector of x, used for training the logistic regression classification model.

    Returns
    -------
    The derivative.

    '''
    N, p = x.shape
    derivative = [0]*p
    for j in range(N):
        dot_product=np.dot(x[j,],wt)
        parenthesis=y[j] - math.exp(dot_product)/(1 + math.exp(dot_product))
        for k in range(p):
            derivative[k] += x[j,k] * parenthesis
    return(derivative)

#update w function
def update_w(wt,derivative,n,l,x):
    '''
    Calculates the new parameter vector w at time t+1. 
    Needs, N (number of observations), n (learning rate), l (shrinkage) to be defined globally.

    Parameters
    ----------
    wt : float
        Vector of parameters for the logistic regression. From beta_0 to beta_p, at time t.
    derivative : float
        Vector of derivatives ued to update the parameter vector wt.
        Obtained from the derivative_fun() function.

    Returns
    -------
    wt at time t+1 iteration.

    '''
    N, p = x.shape
    derivative_term = [i * (n/N) for i in derivative]
    wt_next = wt - n*l*wt + derivative_term
    return(wt_next)

#algorithm function

def logit_algorithm(x,y,n,l,iterations):
    '''
    Runs the logistic regression algorithm.

    Parameters
    ----------
    x : numeric
        Data matrix with N observations and p features, used for training the model.
        A column of ones at index 0 is expected.
    y : binary 0/1
        Label vector of x, used for training the logistic regression classification model.
    n : float
        Learning rate used in the gradient ascent update of wt.
    l : float
        Shrikange value used in the gradient ascent update of wt.
     iterations : integer
         Number of iterations to run the algorithm.

    Returns
    -------
    a list of log-likelihood values for each iteration step.
    a vector of coeficients for the logistic regression that best fit the data.

    '''
    N, p =x.shape
    #Initializing vector w at time zero
    W=np.zeros((p,), dtype=int)
    #Initialize a list to store the likelihood values
    L=[]
    #run the algorithm for a specific number of iterations
    for t in range(iterations):
        Lt=likelihood(W,x,y)
        D = derivative_func(W,x,y)
        W =update_w(W,D,n,l,x)
        L.append(Lt)
        if t % 10 == 0: 
            print("iteration = {}, L = {}".format(t, Lt))
    return(L, W)

#Making predictions function
def y_predict(y,x,W):
    '''
    Makes predictions of the class label for each observation in x and compares with the true labels in y.

    Parameters
    ----------
    y : binary 0/1
        Label vector of x, used for training the logistic regression classification model.
    x : numeric
        Data matrix with N observations and p features, used for training the model.
        A column of ones at index 0 is expected.
    W : float
        Vector of size p with the best fit coeficients obtained running the logistic regression algorithm.
   
    Returns
    -------
    The misclassification error for the predictions.

    '''
    #creating a list to store the predictions
    y_predict = []
    #transforming our prediction probabilities to 0 and 1
    for row in x:
        score=np.dot(W,row)
        prob=math.exp(score) / 1+math.exp(score) #probability that observation in row is 1
        if (prob < 0.5):
            y_predict.append(0)
        else:
            y_predict.append(1)
    #calculating the misclassification error as 1 - accuracy score
    error = 1 - accuracy_score(y,y_predict)
    return(error)