import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt



def get_data():
    diabetes = datasets.load_diabetes()
    x = diabetes.data
    y = diabetes.target
    cases_num = 447
    x = x[:cases_num, :]
    y = y[:cases_num]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, y_train, x_test, y_test


def init_param(n_features, l1):
    param = {}
    param['num_epochs'] = 1000
    param['weights'] = np.random.randn(n_features, 1)
    param['learning_rate'] = 0.8
    param['l1'] = l1
    param['l2'] = 0.0
    return param


def proximal_method(weights_update, param):
    l1 = param['l1']
    l2 = param['l2']
    learning_rate = param['learning_rate']
    sign_weights = np.sign(weights_update)
    abs_weights = np.abs(weights_update)
    softmax_weights = abs_weights - l1 * learning_rate
    proximal_weights = sign_weights * softmax_weights * (softmax_weights > 0)
    proximal_weights /= (1 + learning_rate * l2)
    return proximal_weights



def elastic_net():
    x_train, y_train, x_test, y_test = get_data()
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    n_features = x_train.shape[1]
    l1_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    weights_history = np.zeros((len(l1_list), n_features))
    for i in range(len(l1_list)):
        param = init_param(n_features, l1_list[i])
        num_epochs = param['num_epochs']
        num_cases = x_train.shape[0]
        learning_rate = param['learning_rate']
        print 'start training:'
        for j in range(num_epochs):
            weights = param['weights']
            predict = x_train.dot(weights)
            residual = predict - y_train
            loss = np.square(residual)
            loss = loss.sum() / num_cases / 2
            #print 'epoch %d, loss: %f' % (j, loss)
            gradient = x_train.T.dot(x_train).dot(weights) - x_train.T.dot(y_train)
            gradient = gradient / num_cases
            weights = proximal_method(weights - learning_rate * gradient, param)
            param['weights'] = weights
        print 'training finished'
        weights = param['weights']
        predict = x_test.dot(weights)
        residual = predict - y_test
        loss = np.square(residual)
        loss = loss.sum() / x_test.shape[0] / 2
        weights_history[i, :] = weights[:, 0]
        #print 'test loss is %f' % loss
    plt.figure()
    plot_feature = [0, 1, 4, 5, 6]
    color_feature = ['r', 'g', 'b', 'y', 'c']
    for i in range(5):
        plt.plot(l1_list, weights_history[:, plot_feature[i]], color_feature[i])
    plt.title('Elastic net')
    plt.xlabel('l1')
    plt.ylabel('coefficient')
    plt.show()



if __name__ == '__main__':
    np.random.seed(42)
    elastic_net()
