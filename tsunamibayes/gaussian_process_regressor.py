import numpy as np
from scipy.linalg import solve_triangular


class GPR:
    """Gaussian Process regressors can be instantiated and sampled from
    using this class. The code is based on Algorithm 2.1 from Rasmussen
    and Williams' Gaussian process regression book.
    """
    def __init__(self, kernel=None, noise_level=0.00005):
        """Must provide the kernel function and a noise_level. The kernel
        function should take two inputs and return a covariance matrix
        between those inputs. The noise_level describes how noisy we think
        the input data is. A noise level of 0 will cause the GP to exactly
        interpolate the data points, unless you account for the noise in
        your kernel function.
        """
        # Use the default kernel if not provided.
        if kernel is None: kernel = self.rbf_kernel

        self.kernel = kernel
        self.noise_level = noise_level


    @staticmethod
    def rbf_kernel(x1,x2,sig=1.0):
        """The RBF kernel (squared exponential distance)."""
        print("""We're in gp_Regressor.py in rbf_kernel
        We will print out the x1 array and see how that goes

        ____________________________
        """)
        # print(x1)
        # print(x1[:,np.newaxis])
        # print("x2")
        # print(x2)
        # print("subtracting")
        # print(x1[:,np.newaxis] - x2)
        sqdist = np.linalg.norm(x1[:,np.newaxis] - x2, axis=-1)**2
        return np.exp( -sqdist / (2*sig**2))


    def fit(self, X, y):
        """Trains the Gaussian process by finding the internal parameters K,
        L.
        """
        # Take the output mean so we can normalize our GP's output.
        self.mu_hat = y.mean()

        self.X = X
        self.y = y - self.mu_hat # Now the y's have sample mean 0.
        K = self.kernel(X, X)
        self.L = np.linalg.cholesky(K + self.noise_level*np.eye(len(X)))
        temp = solve_triangular(self.L, self.y, lower=True)
        self.alpha = solve_triangular(self.L.T, temp)


    def predict(self, x_star, return_std=False):
        """Returns the mean of the GP at x_star. Can also return the standard
        deviation at each of those points if desired. Note that the standard
        deviation calculations are the most performance-intensive aspect
        of this function.
        """
        # Compute the mean at the test points.
        k_star = self.kernel(self.X, x_star)
        mu = k_star.T @ self.alpha
        mu += self.mu_hat # So that the output has the correct sample mean.

        # Compute the standard deviation.
        if return_std:
            v = solve_triangular(self.L, k_star, lower=True)
            # We take np.diag in the following line to discard covariances.
            std = np.sqrt(np.diag(self.kernel(x_star, x_star) - v.T @ v))
            return mu, std

        return mu


    def llh(self):
        """Returns the log-likelihood of the trained GPR."""
        llh = -0.5*self.y.T @ self.alpha
        llh -= np.sum(np.log(np.diag(self.L)))
        llh -= len(self.L)/2*np.log(2*np.pi)
        return llh
