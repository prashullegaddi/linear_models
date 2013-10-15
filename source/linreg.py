import numpy as np
import numpy.linalg as la

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy import optimize


# Utils
def read_from_file(infile, delim):
    """ Reads csv from file; last col is the target value and return X, y as
    numpy arrays. """
    data = np.loadtxt(infile, delimiter=delim, dtype=float)
    X = [list(a[:-1]) for a in data]
    y = [a[-1] for a in data]

    ones_to_stack = np.ones((len(y), 1))
    Xn = np.concatenate((ones_to_stack, np.array(X, dtype=float)), axis=1)
    yn = np.array(y, dtype=float)

    return Xn, yn


# Definitions of cost function (obj), and gradient
def objective_function(theta, X, y):
    """ A function to compute the value of objective function
        for parameter theta.
    """
    # m number of training instances
    m = X.shape[0]
    jtheta = sum((np.dot(X, theta) - y)**2) / (2.0*m)
    return jtheta


def compute_gradient(theta, X, y):
    """ Returns the gradient of objection function at theta """
    m = X.shape[0]
    grad_theta = np.dot(X.transpose(), (np.dot(X, theta) - y)) / m
    #print theta, grad_theta, objective_function(theta, X, y)
    return grad_theta


# Andrew Ng's lectures
def linreg_closedform(X, y):
    """ Computes the parameters theta by fitting a hyperplane to the data.  """
    xtx_inv = la.inv(np.dot(X.transpose(), X))
    theta = np.dot(xtx_inv, np.dot(X.transpose(), y))
    return theta


def linreg_batch_grad(X, y, alpha=.01, num_iter=1000):
    """ linear regression: batch gradient method """
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n, dtype=float)

    for itr in range(num_iter):
        cost = objective_function(theta, X, y)
        gradient = compute_gradient(theta, X, y)
        theta = theta - alpha * gradient
        print itr, theta, gradient

    return theta


def linreg_stochastic_grad(X, y, alpha=.01):
    """ Performs linear regression by stochastic gradient method """
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n)
    for i in range(m):
        delta = alpha * (np.dot(theta.transpose(), X[i,:]) -y[i]) * X[i,:]
        theta = theta - delta
    return theta


# Scikit-learn
def linreg_scikit(X, y):
    """ Ordinary least squares regression from scikit-learn """
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    theta = lr.coef_.tolist()[1:]
    theta.insert(0, lr.intercept_)
    return theta


def linreg_gbrt(X, y):
    """ Gradient boosted regression decision trees """
    gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
                max_depth=1, random_state=0, loss='ls')
    gbrt.fit(X, y)
    return gbrt


def linreg_gbrt_predict(gbrt, X):
    return gbrt.predict(X)


# Using scipy optimization methods
def linreg_scipy_opt(X, y, optmethod='ncg'):
    """ Gradient method """
    all_opt_methods = { 'cg': optimize.fmin_cg,
                    'ncg': optimize.fmin_ncg,
                    'bfgs': optimize.fmin_bfgs,
                    'l_bfgs_b': optimize.fmin_l_bfgs_b,
                    'tnc': optimize.fmin_tnc}

    chosen_opt_method = all_opt_methods[optmethod]

    n = X.shape[1]
    theta = np.zeros(n)
    result = chosen_opt_method(objective_function, theta,
                    compute_gradient, args=(X, y))
    return result


def predict(theta, X):
    return np.dot(X, theta)


def demo(trainfile, delim=','):
    X, y = read_from_file(trainfile, delim)

##    print 'Closed form:'
##    print linreg_closedform(X, y)
##    print 'Batch gradient:'
##    print linreg_batch_grad(X, y)
##    print 'Stochastic gradient:'
##    print linreg_stochastic_grad(X, y)
    print 'Scikit '
    theta1 = linreg_scikit(X, y)
    print 'NormalLR: MSE on train: ', mean_squared_error(y, predict(theta1, X))
    print 'GBRT'
    gbrt_model = linreg_gbrt(X, y)
    print 'GBRT: MSE on train: ', mean_squared_error(y, linreg_gbrt_predict(gbrt_model, X))


def lowess(X, y):
    pass


def compare():
    """ Different regression algorithms compared """
    from sklearn import datasets
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error

    boston_data = datasets.load_boston()
    X, y = boston_data.data, boston_data.target
    linreg = linear_model.LinearRegression()
    lr_model = linreg.fit(X ,y)
    lr_mse = mean_squared_error(lr_model.predict(X), y)
    print 'Linear regression:', lr_mse

    ridge = linear_model.Ridge()
    ridge_model = ridge.fit(X, y)
    ridge_mse = mean_squared_error(ridge.predict(X), y)
    print 'Ridge regression:', ridge_mse

    lasso = linear_model.Lasso()
    lasso_model = lasso.fit(X, y)
    lasso_mse = mean_squared_error(lasso_model.predict(X), y)
    print 'Lasso regression:', lasso_mse


if __name__ == '__main__':
    demo('ex1data1.txt')


