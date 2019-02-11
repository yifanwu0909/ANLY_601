import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Compute Gaussian Likelihoods
def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))


def plt_2dGaussians(mu0, mu1, sig0, sig1):
    delta = 0.025
    x = np.arange(-2.0, 4.0, delta)
    y = np.arange(-1.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = gaussian_2d(X, Y, mu0[0], mu0[1], sig0[0], sig0[1])
    Z2 = gaussian_2d(X, Y, mu1[0], mu1[1], sig1[0], sig1[1])

    # Create a contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    
    # For use help, see https://matplotlib.org/
    plt.clf()
    plt.figure(figsize=(10,5))
    CS1 = plt.contour(X, Y, Z2)
    plt.clabel(CS1, inline=1, fontsize=10)
    CS2 = plt.contour(X, Y, Z1)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.scatter(X1, X2)
    
    plt.title('Learned Gaussian Contours')
    plt.show()
       
    
def EM(X, m = 2, epsilon = 0.000001):
    # let's assume dim l = 2

    theta_pre = np.random.random(4*m)
    mu = theta_pre[0:2*m]
    sigma = theta_pre[2*m:4*m]
    
    if m == 2:
        plt_2dGaussians(mu[0:2], mu[2:4], sigma[0:2], sigma[2:4])
    
    N = X.shape[0]
    prior_track = []
    mu_track = []
    sigma_track = []
    
    prior = np.array([1/m]*m)

    f = np.zeros((m, N))
    for k in range(m):
        f[k] = multivariate_normal.pdf(X, mean=mu[2*k:2*(k+1)], cov=sigma[2*k:2*(k+1)])
    
    T = np.zeros((m, N))
    for k in range(m):
        T[k] = prior[k]*f[k]/np.matmul(prior, f)
    
    mu_after = np.zeros(2*m)
    sigma_after = np.zeros(2*m)
    
    for k in range(m):
        mu_after[2*k:2*(k+1)] = np.sum(T[k].reshape(N,1)*X, axis = 0)/np.sum(T[k])
        sigma_after[2*k:2*(k+1)] = np.sqrt(np.sum(T[k].reshape(N,1)*(X-mu_after[2*k:2*(k+1)])**2, axis = 0)/np.sum(T[k]))
    
    if m == 2:
        plt_2dGaussians(mu_after[0:2], mu_after[2:4], sigma_after[0:2], sigma_after[2:4])
    
    theta = np.concatenate([mu_after, sigma_after])
    
    for k in range(m):
        prior[k] = np.sum(T[k])/N
        
    prior_track.append(prior) 
    mu_track.append(mu_after)
    sigma_track.append(sigma_after)
    
    i = 1
    
    while max(abs(theta-theta_pre)) > epsilon:
        print("now iteration " + str(i))
        theta_pre = theta
        mu = theta_pre[0:2*m]
        sigma = theta_pre[2*m:4*m]
        
        for k in range(m):
            f[k] = multivariate_normal.pdf(X, mean=mu[2*k:2*(k+1)], cov=sigma[2*k:2*(k+1)])
        
        for k in range(m):
            T[k] = prior[k]*f[k]/np.matmul(prior, f)
        
        mu_after = np.zeros(2*m)
        sigma_after = np.zeros(2*m)
        for k in range(m):
            mu_after[2*k:2*(k+1)] = np.sum(T[k].reshape(N,1)*X, axis = 0)/np.sum(T[k])
            sigma_after[2*k:2*(k+1)] = np.sqrt(np.sum(T[k].reshape(N,1)*(X-mu_after[2*k:2*(k+1)])**2, axis = 0)/np.sum(T[k]))
        print(mu_after)
        if m == 2:
            plt_2dGaussians(mu_after[0:2], mu_after[2:4], sigma_after[0:2], sigma_after[2:4])
        
        theta = np.concatenate([mu_after, sigma_after])
        
        prior = np.zeros(m)
        for k in range(m):
            prior[k] = np.sum(T[k])/N
            
        i += 1
        prior_track.append(prior)    
        mu_track.append(mu_after)
        sigma_track.append(sigma_after)
        
    print('Total iterations: ', i)
    
    return mu_after, sigma, prior, mu_track


X1 = [ 1, 1.5, 1.2, 1.2, .9, .8, 1, 2.3, 2.1, 2, 3, 2.5, 3]
X2 = [ 1.5, 1.2, 1.2, .9, .7, .8, 2.3, 2.1, 2, 3, 2.5, 3, 2.1]  
X = np.column_stack([X1, X2])

mu, sigma, prior, mu_track = EM(X, m = 2, epsilon = 0.000001)
