import numpy as np
import numpy.linalg as la

from sklearn import linear_model
from scipy import optimize


def read_from_file(infile, delim):
    """ Reads csv from file; last col is the target value and return X, y as
    numpy arrays. """
    data = np.loadtxt(infile, delimiter=delim, dtype=float)
    X = [list(a[:-1]) for a in data]
    y = [a[-1] for a in data]

    ones_to_stack = np.ones((len(y), 1))
    Xn = np.concatenate((ones_to_stack, np.array(X)), axis=1)
    yn = np.array(y)

    return Xn, yn


# Andrew Ng's lectures
def linreg_closedform(X, y):
    """ Computes the parameters theta by fitting a hyperplane to the data.  """
    xtx_inv = la.inv(np.dot(X.transpose(), X))
    theta = np.dot(xtx_inv, np.dot(X.transpose(), y))
    return theta

def linreg_stochastic_grad(X, y, alpha=.5):
    """ Performs linear regression by stochastic gradient method """
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n)




# Scikit-learn
def linreg_scikit(X, y):
    """ Ordinary least squares regression from scikit-learn """
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    return lr.intercept_, lr.coef_



# Using scipy optimization methods
def linreg_gradient(X, y, optmethod='ncg'):
    """ Gradient method """
    all_opt_methods = { 'cg': optimize.fmin_cg,
                    'ncg': optimize.fmin_ncg,
                    'bfgs': optimize.fmin_bfgs,
                    'l_bfgs_b': optimize.fmin_l_bfgs_b,
                    'tnc': optimize.fmin_tnc}

    chosen_opt_method = all_opt_methods[optmethod]

    def objective_function(theta, X, y):
        """ A function to compute the value of objective function
            for parameter theta.
        """
        # m number of training instances
        m = X.shape[0]
        jtheta = sum((np.dot(X, theta) - y)**2) / 2.0
        return jtheta

    def gradient(theta, X, y):
        """ Returns the gradient of objection function at theta """
        grad_theta = np.dot(X.transpose(), (np.dot(X, theta) - y))
        return grad_theta

    n = X.shape[1]
    theta = np.zeros(n)
    result = chosen_opt_method(objective_function, theta, gradient, args=(X, y))
    return result