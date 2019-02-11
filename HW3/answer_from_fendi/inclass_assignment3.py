# -*- coding: utf-8 -*-
"""


In Class Exercise:  xx/100  (+ 10 extra credit if submitted before end of day of class)
******************************************************

Goal: Implement EM, for Gaussian Mixture given data X.
mu, sigma, prior = EM(X, m, epsilon)

mu is lxm where l is dimensionality and m is number of mixture components
sigma is m (diagonal) covariance matrices (so really just m vectors), eg (lxm)
prior is prior for each component 1xm (one by m)
epsilon is the total change in updated parameter values and is used as a stopping criterion

******************************************************

The functions should compute Gaussian Mixture parameters using EM given data set X. Plot
the results of the learned parameters (using a contour like display given below) after 
each iteration (or so) of the learning process. This will help to visualize the 
process of learning. Feel free  to use the code snippets provided below
or feel free to edit as you see fit. Assume covariance matrices are diagonal. 

Construct synthetic 2-d data X or find a simple data set to use to test your code.
********************************************************




Take-Home Assignment:  xx/100
******************************************************
To Do: Type-up responses and supporting figures and submit as a PDF. 
******************************************************

#1 Using EM algorithm to derive the update equation for the mean parameter for
each Gaussian Mixture Compoenent j. 

#2 Assume data set as follows:

x[0] = [ 1, 1.5, 1.2, 1.2, .9, .8, 1, 2.3, 2.1, 2, 3, 2.5, 3]                                   
x[1] = [ 1.5, 1.2, 1.2, .9, .7, .8, 2.3, 2.1, 2, 3, 2.5, 3, 2.1]     

a) Run your code with 1 component, 2 components, and 3 components. What epsilon did you use? 
Which component number "fits" the data best? How did you make this determination?

b) To your existing code, add a function that plots (in a different figure) 
the loglikelihood as a function  of iteration. Can you use this information as
a stopping criterion rather than epsilon? Can you use this information to 
determine which number of components is best?    

c) Given your analysis in Part b), what was the learned parameters of the mixture
model that best fit the data

********************************************************

For supplemental help ...
See Resources: 
    https://matplotlib.org/
    [TK], [Bp], [DHS]


@author: jerem
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os
from scipy.misc import imread
import math

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider 


# Compute Gaussian Likelihoods
def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))*(1/((2 *math.pi)* (xsig*ysig)**(1/2)))

def obj_function(sig,mu,X,m,px_2d,p0,pj_2d):
    q = 0
    for i in range(len(X)):
        for j in range(m):
            q = q+ pj_2d[j][i]*np.log(px_2d[j][i]*p0[j])
    return q
                 
def EM(X, m, epsilon):
    # initialize mu0, sig0 and p0
#    mu0 = [[0,0.2],[0.23,0.42]]
#    sig0 = [[1.2,0.2],[1.23,1.42]]
    mu0 = [np.random.uniform(-10,10,2) for i in range(m)]
    sig0 = [np.random.uniform(0.5,15,2) for i in range(m)]
    p0 = [1/m for i in range(m)]
    x = X[:,0]
    y = X[:,1]
    px_2d = []
#    print(mu0)
    count = 0
#    print(sig0)
    while True:
        count = count+1
        for i in range(m):
            px_2d.append(gaussian_2d(x,y,mu0[i][0],mu0[i][1],sig0[i][0],sig0[i][1]))
        px_theta_2d = [0 for i in range(len(X))]
        for j in range(m):  
            px_theta_2d = px_theta_2d + px_2d[j]*p0[j]
        pj_2d = []
        for i in range(m):
            pj_2d.append(px_2d[i]*p0[i]/px_theta_2d)
        old_obj_val = obj_function(sig0,mu0,X,m,px_2d,p0,pj_2d)
        # update p0
        for j in range(m):
            p0[j] = pj_2d[j].sum()/len(X)
        # update sigma
        for i in range(m):
            sig0[i] = [((pj_2d[i]*(x-mu0[i][0])**2).sum()/(pj_2d[i].sum()*2)),
                ((pj_2d[i]*(x-mu0[i][1])**2).sum()/(pj_2d[i].sum()*2))]
        # update mu
        for i in range(m):
            mu0[i] = [(pj_2d[i]*x).sum()/(pj_2d[i].sum()),(pj_2d[i]*y).sum()/(pj_2d[i].sum())]
        new_obj_val = obj_function(sig0,mu0,X,m,px_2d,p0,pj_2d)
        
        fig1 = plt.ioff()
        plt.figure(figsize=(10,5))
        plt.clf()
        # Create appropriate grid for display purposes.
        delta = 0.025
        x_plt = np.arange(-3.0, 3.0, delta)
        y_plt = np.arange(-2.0, 2.0, delta)
        X_plt, Y_plt = np.meshgrid(x_plt, y_plt)

        Z1 = gaussian_2d(X_plt, Y_plt, mu0[0][0], mu0[0][1], sig0[0][0],sig0[0][1])
        #Z2 = gaussian_2d(X, Y, mu1[0], mu1[1],  sig1[0],sig1[1])
        CS1 = plt.contour(X_plt, Y_plt, Z1)
        plt.title('Learned Gaussian Contours after'+str(count)+'th iteration' )
        plt.clabel(CS1, inline=1, fontsize=10)
        #plt.clabel(CS2, inline=1, fontsize=10)
        plt.pause(1)
        

        # Code to create a figure and repeatedly plot the newly learned mixture.
    
        
        if np.abs(old_obj_val - new_obj_val)< epsilon:
            return [mu0, sig0, p0]
        
m = 2
epsilon = 0.0001
X = (np.random.random((20,2))-0.5)*3
mu, sigma, prior = EM(X, m, epsilon)

