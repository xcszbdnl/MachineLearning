import numpy as np
from sklearn import gaussian_process
from matplotlib import pyplot as plt
import math

GRADIENT_THRESHOLD = 1e-2

def f(x):
    return x * np.sin(x)


def standard_process():
    """
    standard process from scikit learn
    """
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(X, y)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig = plt.figure()
    plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


def kernel(x_1, x_2, sigma, l):
    result = (x_1 - x_2) ** 2
    result /= (2 * (l ** 2))
    result = -result
    result = np.exp(result)
    result *= (sigma ** 2)
    return result


def covariance(x_1, x_2, sigma, l):
    num_cases1 = len(x_1)
    num_cases2 = len(x_2)
    cov = np.zeros((num_cases1, num_cases2))
    for i in range(num_cases1):
        for j in range(num_cases2):
            cov[i, j] = kernel(x_1[i, 0], x_2[j, 0], sigma, l)
    return cov


def hyperparam_likelihood(x_train, y_train):
    learning_rate = 0.1
    sigma = 1.0
    l = 2.0
    gradient_l = np.inf
    gradient_sigma = np.inf
    num_epoch = 200
    cnt_epoch = 0
    num_cases = x_train.shape[0]
    while cnt_epoch < num_epoch and (abs(gradient_l) > GRADIENT_THRESHOLD or abs(gradient_sigma) > GRADIENT_THRESHOLD):
        cov = covariance(x_train, x_train, sigma, l)
        cov_inv = np.linalg.inv(cov)
        temp = cov * 2 / sigma
        temp = cov_inv.dot(temp)
        gradient_sigma = -0.5 * np.trace(temp)
        r = y_train.T.dot(temp)
        r = r.dot(cov_inv)
        r = r.dot(y_train)
        gradient_sigma += 0.5 * r
        sigma += learning_rate * gradient_sigma

        temp = cov
        for i in range(num_cases):
            for j in range(num_cases):
                temp[i, j] *= ((x_train[i, 0] - x_train[j, 0]) / math.pow(l, 3))

        temp = cov_inv.dot(temp)
        gradient_l = -0.5 * np.trace(temp)
        r = y_train.T.dot(temp)
        r = r.dot(cov_inv)
        r = r.dot(y_train)
        gradient_l += 0.5 * r
        sigma += learning_rate * gradient_sigma
        l += learning_rate * gradient_l
        cnt_epoch += 1
        print 'epoch %d, gradient sigma: %g, gradient l: %g, sigma: %g, l: %g\n' % (cnt_epoch, gradient_sigma, gradient_l, sigma, l)
    return sigma, l


def getData():
    x_train = np.array([1., 3., 5., 6., 7., 8.])
    y_train = f(x_train).ravel()
    x_test = np.linspace(0, 10, 1000)
    x_train = x_train.reshape(x_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    return x_train, y_train, x_test


def gaussian_process_test():
    x_train, y_train, x_test = getData()
    sigma, l = hyperparam_likelihood(x_train, y_train)
    K = covariance(x_train, x_train, sigma, l)
    K_star = covariance(x_test, x_train, sigma, l)
    K_star_star_ = covariance(x_test, x_test, sigma, l)
    K_inv = np.linalg.inv(K)
    predict = K_star.dot(K_inv)
    predict = predict.dot(y_train)
    fig = plt.figure()
    plt.plot(x_test, f(x_test), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(x_test, predict, 'b-', label=u'Prediction')
    plt.plot(x_train, y_train, 'r.', markersize=10, label=u'Observations')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # standard_process()
    gaussian_process_test()
