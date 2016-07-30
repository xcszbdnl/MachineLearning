import numpy as np
import matplotlib.pyplot as plt
from random import randrange


def get_data():
    data_path = '../Data/mnist.npz'
    data_set = np.load(data_path)
    return data_set['x_train'], data_set['y_train'], data_set['x_test'], data_set['y_test']


def mlp_setup(node_number):
    net = {}
    net['node_number'] = node_number
    net['weights'] = []
    for i in range(1, len(node_number)):
        cnt_weight = (np.random.randn(node_number[i], node_number[i - 1] + 1) - 0.5) * 2 * 4 * np.sqrt(6.0 / (node_number[i] + node_number[i - 1]))
        net['weights'].append(cnt_weight)
    net['batch_size'] = 100
    net['num_epoch'] = 100
    net['layer_num'] = len(node_number)
    net['learning_rate'] = 0.1
    net['weight_decay'] = 1e-3
    return net


def mlp_forward(net, x_train):
    net['output'] = []
    num_case = x_train.shape[0]
    x_train = np.hstack((np.ones((num_case, 1)), x_train))
    net['output'].append(x_train)
    layer_num = net['layer_num']
    for i in range(1, layer_num - 1):
        cnt_output = net['output'][i - 1].dot(net['weights'][i - 1].T)
        cnt_output = np.hstack((np.ones((num_case, 1)), cnt_output))
        net['output'].append(cnt_output)
    final_output = net['output'][layer_num - 2].dot(net['weights'][layer_num - 2].T)
    final_output -= np.max(final_output, axis=1)
    final_output = np.exp(final_output)
    final_output /= np.sum(final_output, axis=1)
    net['output'].append(final_output)


def mlp_backward(net, y_train):
    net['delta'] = []
    net['gradient'] = []
    layer_num = net['layer_num']
    for i in range(layer_num):
        net['delta'].append(np.zeros(net['output'][i].shape))
        if i != layer_num - 2:
            net['gradient'].append(np.zeros(net['weights'][i].shape))
    net['delta'][layer_num - 1] = net['output'][layer_num - 1] - y_train
    for i in range(layer_num - 2, 1, step=-1):
        sigmoid_gradient = net['output'][i] * (1 - net['output'][i])
        if i == layer_num - 2:
            net['delta'][i] = net['delta'][i + 1].dot(net['weights'][i]) * sigmoid_gradient
        else:
            net['delta'][i] = (net['delta'][i + 1][:, 2:]).dot(net['weights'][i]) * sigmoid_gradient
    for i in range(0, layer_num - 1):
        if i == layer_num - 2:
            net['gradient'][i] = net['delta'][i + 1].T.dot(net['output'][i]) / net['batch_size']
        else:
            net['gradient'][i] = net['delta'][i + 1][:, 2:].T.dot(net['output'][i]) / net['batch_size']


def mlp_apply_grad(net):
    layer_num = net['layer_num']
    for i in range(layer_num - 1):
        dw = net['gradient'][i] + np.hstack((np.zeros((net['weights'][i].shape[0], 1)), net['weights'][i][:, 2:]))
        dw = net['learning_rate'] * dw
        net['weights'][i] = net['weights'][i] - dw


def mlp_test(net, x_test, y_test):
    mlp_forward(net, x_test)
    predict = net['output'][net['layer_num'] - 1]
    label = np.argsort(-predict, axis=1)[:,0]
    err_num = 0
    for i in range(len(x_test)):
        if y_test[label[i]] == 0:
            err_num += 1
    return err_num


def mlp_train(net, x_train, y_train):
    num_case = x_train.shape[0]
    for i in range(net['num_epoch']):
        tot_error = 0
        random_index = np.random.choice(num_case, num_case)
        batch_size = net['batch_size']
        for l in range(num_case / batch_size):
            cnt_batch = x_train[random_index[l * batch_size : (l + 1) * batch_size], :]
            cnt_target = y_train[random_index[l * batch_size : (l + 1) * batch_size], :]
            mlp_forward(net, cnt_batch)
            mlp_backward(net, batch_size, cnt_target)
            mlp_apply_grad(net)
            error_num = mlp_test(net, cnt_batch, cnt_target)
            tot_error += error_num
        print 'epoch %d, classification error :%f' % (i, tot_error * 1.0 / num_case)
    return net


def mlp():
    x_train, y_train, x_test, y_test = get_data()
    x_train = x_train.astype(float)
    x_train /= 255.0

    x1 = x_train[0, :]
    x1 = np.reshape(x1, (28, 28))
    plt.figure(1, (12, 6))
    plt.imshow(x1, plt.gray())
    plt.show()

    np.random.seed(42)
    net = mlp_setup([784, 100, 10])
    net = mlp_train(net, x_train, y_train)
    error_num = mlp_test(net, x_test, y_test)
    print 'after trained %d misclassification' % error_num



if __name__ == '__main__':
    mlp()
