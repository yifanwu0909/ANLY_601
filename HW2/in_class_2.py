from sklearn import svm, datasets
import numpy as np


def gaussianML(X):
    mu0 = X.mean(axis = 0)
    sigma_matrix = 0
    for i in range(len(X)):
        gap = np.array([X[i] - mu0])
        sigma_matrix = sigma_matrix + np.matmul(gap.T, gap)
    sigma0 = (sigma_matrix/len(X)).diagonal().tolist()
    return mu0, sigma0
    
    
def gaussianMAP(X, priorMu, sigmaGuess):
    sigma0_0 = 0.1
    sigma0_1 = 0.1
    mu_0 = (priorMu[0] + ((sigma0_0**2.0)/(sigmaGuess[0]**2.0)) * sum(X[:, 0]))/(1 + ((sigma0_0**2.0)/(sigmaGuess[0]**2.0) * len(X[:, 0])))
    mu_1 = (priorMu[1] + ((sigma0_1**2.0)/(sigmaGuess[1]**2.0)) * sum(X[:, 1]))/(1 + ((sigma0_1**2.0)/(sigmaGuess[1]**2.0) * len(X[:, 1])))
    mu = [mu_0, mu_1]
    return mu, sigmaGuess

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))


def plt_2dGaussians(mu0, mu1, sig0, sig1):
    delta = 0.025
    x = np.arange(3.0, 8.0, delta)
    y = np.arange(1.0, 6.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = gaussian_2d(X, Y, mu0[0], mu0[1], sig0[0],sig0[1])
    Z2 = gaussian_2d(X, Y, mu1[0], mu1[1],  sig1[0],sig1[1])

    # Create a contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    
    # For use help, see https://matplotlib.org/
    plt.clf()
    plt.figure(figsize=(20,10))
    CS1 = plt.contour(X, Y, Z2)
    plt.clabel(CS1, inline=1, fontsize=10)
    CS2 = plt.contour(X, Y, Z1)
    plt.clabel(CS2, inline=1, fontsize=10)
    
    plt.title('Learned Gaussian Contours')


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

mu0, sigma0 = gaussianML(X)

sigmaGuess = [1, 1]
priorMu = [3, 5] 
mu1, sigmaGuess = gaussianMAP(X, priorMu, sigmaGuess)

plt_2dGaussians(mu0,mu1, sigma0, sigmaGuess)
