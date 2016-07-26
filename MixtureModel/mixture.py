import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
from sklearn import mixture

ITER_STEP = 100
LOG_THRESHOLD = -100
TOT_COMPONENT = 3

def generate_data():
    n_samples = 300
    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

    normal_gaussian = np.random.randn(n_samples, 2)

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C) + np.array([10, 10])

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian, normal_gaussian])

    return X_train

def standard_model():
    X_train = generate_data()
    # fit a Gaussian Mixture Model with three components
    clf = mixture.GMM(n_components=TOT_COMPONENT, covariance_type='full')
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)[0]
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()


def init_parameter():
    param = {}
    param['pie'] = np.ones(TOT_COMPONENT) * 1.0 / TOT_COMPONENT
    param['mean'] = np.random.randn(TOT_COMPONENT, 2)

    param['mean'][0] = np.array([0, 0])
    param['mean'][1] = np.array([10, 10])
    param['mean'][2] = np.array([20, 20])

    param['precision'] = np.zeros((TOT_COMPONENT, 2, 2))
    param['covariance'] = np.zeros((TOT_COMPONENT, 2, 2))
    for i in range(TOT_COMPONENT):
        param['precision'][i] = np.eye(2, 2)
        param['covariance'][i] = np.eye(2,2)
    return param


def density(x, mu, covariance, precision):
    result = 1.0 / (2 * math.pi)
    result = result / np.sqrt(np.linalg.det(precision))
    z = (x - mu).reshape(2, 1)
    result = result * np.exp(-1.0 / 2 * z.T.dot(covariance).dot(z))
    return result


def E_step(X_train, param):
    z_nk = np.zeros((X_train.shape[0], TOT_COMPONENT))
    for i in range(X_train.shape[0]):
        sum = 0
        for j in range(TOT_COMPONENT):
            z_nk[i][j] = param['pie'][j] * density(X_train[i, :], param['mean'][j], param['covariance'][j], param['precision'][j])
            sum += z_nk[i][j]
        for j in range(TOT_COMPONENT):
            z_nk[i][j] /= sum

    param['z_nk'] = z_nk

def M_step(X_train, param):
    for i in range(TOT_COMPONENT):
        mean = np.zeros((1, X_train.shape[1]))
        denominator = 0.0
        precision = np.zeros(param['precision'][i].shape)
        for j in range(X_train.shape[0]):
            denominator += param['z_nk'][j][i]
            mean += param['z_nk'][j][i] * X_train[j, :]
            z = (X_train[j, :] - param['mean'][i]).reshape(2, 1)
            precision += param['z_nk'][j][i] * z.dot(z.T)
        param['mean'][i] = mean / denominator
        param['pie'][i] = denominator / X_train.shape[0]
        param['precision'][i] = precision / denominator
        param['covariance'][i] = np.linalg.inv(param['precision'][i])


def loglikehood(X_train, param):
    result = 0.0
    for i in range(X_train.shape[0]):
        cnt_log = 0.0
        for j in range(TOT_COMPONENT):
            cnt_log += param['pie'][j] * density(X_train[i, :], param['mean'][j], param['covariance'][j], param['precision'][j])
        cnt_log = np.log2(cnt_log)
        result += cnt_log
    return result

def mixture_model():
    X_train = generate_data()
    step = 1
    param = init_parameter()
    data_log = -np.inf
    while (data_log < LOG_THRESHOLD and step < ITER_STEP):
        data_log = loglikehood(X_train, param)
        print 'STEP %d\'s loglikelihood is %f' % (step, data_log)
        E_step(X_train, param)
        M_step(X_train, param)
        step += 1
    Y_label = np.argsort(-param['z_nk'], axis=1)
    plt.figure(1, (12, 6))
    x1 = X_train[Y_label[:, 0] == 0, :]
    x2 = X_train[Y_label[:, 0] == 1, :]
    x3 = X_train[Y_label[:, 0] == 2, :]
    plt.scatter(x1[:, 0], x1[:, 1], color='r')
    plt.scatter(x2[:, 0], x2[:, 1], color='b')
    plt.scatter(x3[:, 0], x3[:, 1], color='y')

    plt.show()

if __name__ == '__main__':
    standard_model()
    mixture_model()
