import numpy as np
import matplotlib.pyplot as plt
import copy
from random import randrange


def get_data():
    data_path = '../Data/mnist.npz'
    data_set = np.load(data_path)
    return data_set['x_train'], data_set['y_train'], data_set['x_test'], data_set['y_test']


def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)
    #return 1 / (1 + np.exp(-x))


def mlp_setup(node_number):
    net = {}
    net['node_number'] = node_number
    net['weights'] = []
    for i in range(1, len(node_number)):
        cnt_weight = (np.random.rand(node_number[i], node_number[i - 1] + 1) - 0.5) * 2 * 4 * np.sqrt(6.0 / (node_number[i] + node_number[i - 1]))
        net['weights'].append(cnt_weight)
    net['batch_size'] = 100
    net['num_epoch'] = 100
    net['layer_num'] = len(node_number)
    net['learning_rate'] = 0.1
    net['weight_decay'] = 1e-3
    return net


def mlp_forward(net, x_train, y_train):
    net['output'] = []
    num_case = x_train.shape[0]
    x_train = np.hstack((np.ones((num_case, 1)), x_train))
    net['output'].append(x_train)
    layer_num = net['layer_num']
    for i in range(1, layer_num - 1):
        cnt_output = net['output'][i - 1].dot(net['weights'][i - 1].T)
        cnt_output = np.hstack((np.ones((num_case, 1)), cnt_output))
        vector_sigmoid = np.vectorize(sigmoid)
        cnt_output = vector_sigmoid(cnt_output)
        #cnt_output = sigmoid(cnt_output)
        net['output'].append(cnt_output)
    final_output = net['output'][layer_num - 2].dot(net['weights'][layer_num - 2].T)
    final_output -= np.max(final_output, axis=1).reshape((num_case, 1))
    final_output = np.exp(final_output)
    final_output /= np.sum(final_output, axis=1).reshape((num_case, 1))
    net['output'].append(final_output)
    net['cross_error'] = -(y_train * np.log(final_output)).sum() / num_case


def mlp_backward(net, y_train):
    net['delta'] = []
    net['gradient'] = []
    layer_num = net['layer_num']
    for i in range(layer_num):
        net['delta'].append(np.zeros(net['output'][i].shape))
        if i != layer_num - 1:
            net['gradient'].append(np.zeros(net['weights'][i].shape))
    net['delta'][layer_num - 1] = (net['output'][layer_num - 1] - y_train)
    for i in range(layer_num - 2, 0, -1):
        sigmoid_gradient = net['output'][i] * (1 - net['output'][i])
        if i == layer_num - 2:
            net['delta'][i] = net['delta'][i + 1].dot(net['weights'][i]) * sigmoid_gradient
        else:
            net['delta'][i] = (net['delta'][i + 1][:, 1:]).dot(net['weights'][i]) * sigmoid_gradient
    for i in range(0, layer_num - 1):
        if i == layer_num - 2:
            net['gradient'][i] = net['delta'][i + 1].T.dot(net['output'][i]) / net['batch_size']
        else:
            net['gradient'][i] = net['delta'][i + 1][:, 1:].T.dot(net['output'][i]) / net['batch_size']


def gradient_numerical_check(net, x_train, y_train):
    epsilon = 1e-6
    error_threshold = 1e-7
    layer_num = net['layer_num']
    for l in range(layer_num - 1):
        for i in range(net['weights'][l].shape[0]):
            for j in range(net['weights'][l].shape[1]):
                net_p = copy.deepcopy(net)
                net_n = copy.deepcopy(net)
                net_p['weights'][l][i][j] += epsilon
                net_n['weights'][l][i][j] -= epsilon
                mlp_forward(net_p, x_train, y_train)
                mlp_forward(net_n, x_train, y_train)
                delta = (net_p['cross_error'] - net_n['cross_error']) / (2 * epsilon)
                err = np.abs(delta - net['gradient'][l][i][j])
                if err > error_threshold:
                    print 'gradient calculate error at layer %d, index: (%d, %d)' % (l, i, j)


def mlp_apply_grad(net):
    layer_num = net['layer_num']
    for i in range(layer_num - 1):
        dw = net['gradient'][i] + net['weight_decay'] * np.hstack((np.zeros((net['weights'][i].shape[0], 1)), net['weights'][i][:, 1:]))
        dw = net['learning_rate'] * dw
        net['weights'][i] = net['weights'][i] - dw


def mlp_predict(net, x_test, y_test):
    mlp_forward(net, x_test, y_test)
    predict = net['output'][net['layer_num'] - 1]
    label = np.argsort(-predict, axis=1)[:, 0]
    err_num = 0
    for i in range(len(x_test)):
        if y_test[i, label[i]] == 0:
            err_num += 1
    return err_num


def mlp_train(net, x_train, y_train):
    num_case = x_train.shape[0]
    for i in range(net['num_epoch']):
        tot_error = 0
        tot_cross_error = 0
        random_index = np.random.permutation(num_case)
        batch_size = net['batch_size']
        print 'epoch %d' % i,
        for l in range(num_case / batch_size):
            cnt_batch = x_train[random_index[l * batch_size : (l + 1) * batch_size], :]
            cnt_target = y_train[random_index[l * batch_size : (l + 1) * batch_size], :]
            mlp_forward(net, cnt_batch, cnt_target)
            mlp_backward(net, cnt_target)
            mlp_apply_grad(net)
            error_num = mlp_predict(net, cnt_batch, cnt_target)
            tot_error += error_num
            tot_cross_error += net['cross_error']
        print 'classification error :%f, cross_entropy error:%f' % (tot_error * 1.0 / num_case, tot_cross_error)


def mlp_test():
    x_train, y_train, x_test, y_test = get_data()
    x_train = x_train.astype(float)
    x_train /= 255.0
    """
    x1 = x_train[0, :]
    x1 = np.reshape(x1, (28, 28))
    plt.figure(1, (12, 6))
    plt.imshow(x1, plt.gray())
    plt.show()
    """
    np.random.seed(42)
    net = mlp_setup([784, 100, 10])
    mlp_train(net, x_train, y_train)
    error_num = mlp_predict(net, x_test, y_test)
    print 'after trained %d misclassification' % error_num


def mlp_check_gradient():
    x_train = np.random.rand(20, 5)
    y_train = np.random.rand(20, 2)
    y_train = y_train == np.repeat(np.max(y_train, axis=1).reshape(y_train.shape[0], 1), y_train.shape[1], axis=1)
    y_train = y_train.astype('int')
    net = mlp_setup([5, 4, 2])
    mlp_forward(net, x_train, y_train)
    mlp_backward(net, y_train)
    gradient_numerical_check(net, x_train, y_train)


if __name__ == '__main__':
    np.random.seed(0)
    #mlp_check_gradient()
    mlp_test()
