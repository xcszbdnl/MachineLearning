import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

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

def cnn_setup(net_architecture, x_train, y_train):
    net = {}
    net['architecture'] = net_architecture
    net['weights'] = []
    net['bias'] = []
    net['epoch'] = 100
    net['batch_size'] = 100
    layer_num = len(net_architecture)
    net['layer_num'] = layer_num + 1
    input_map = net_architecture[0]['input_maps']
    map_size = np.array([x_train.shape[0], x_train.shape[1]])
    for l in range(layer_num - 1):
        net['weights'].append([])
        net['bias'].append([])
        if net_architecture[l]['type'] == 'convolutonal':
            kernel_size = net_architecture['kernel_size']
            map_size = map_size - kernel_size + 1
            fan_out = net_architecture[l]['output_maps'] * (kernel_size ** 2)
            for i in range(net_architecture[l]['output_maps']):
                net['weights'][l].append([])
                net['bias'][l].append(0)
                fan_in = input_map * (kernel_size ** 2)
                for j in range(input_map):
                    net['weights'][l][i].append(((np.random.rand(kernel_size, kernel_size) - 0.5) *
                                              2 * np.sqrt(6.0 / (fan_in + fan_out))))
            input_map = net['output_maps']
        elif net_architecture[l]['type'] == 'pooling':
            map_size = map_size / net_architecture['scale']

    final_num = np.prod(map_size) * input_map
    output_num = y_train.shape[1]
    net['weights'].append((np.random.rand(output_num, final_num) - 0.5) * 2 * np.sqrt(6.0 / (final_num + output_num)))
    return net


def cnn_forward(net, x_train, y_train):
    net_architecture = net['architecture']
    net['output'] = []
    net['output'].append([])
    net['output'][0].append(x_train)
    num_cases = x_train.reshape[2]
    layer_num = net['layer_num']
    input_maps = net_architecture[0]['input_map']
    for l in range(1, layer_num - 1):
        net['output'].append([])
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
            kernel_size = net_architecture[l]['kernel_size']
            for i in range(output_maps):
                map_size = net['output'][l - 1][0].shape - (kernel_size - 1, kernel_size - 1, 0)
                result = np.zeros(map_size)
                for j in range(input_maps):
                    result += signal.convolve2d(net['output'][l - 1][j], net['weights'][l][j][i], 'valid')
                result += net['bias'][l][i]
                vector_sigmoid = np.vectorize(sigmoid)
                result = vector_sigmoid(result)
            net['output'][l].append(result)
            input_maps = output_maps
        elif net_architecture[l]['type'] == 'pooling':
            scale = net_architecture[l]['scale']
            map_size /= scale
            mean_pooling = np.ones((scale, scale)) / (scale ** 2)
            for i in range(output_maps):
                result = signal.convolve2d(net['output'][l - 1][j], mean_pooling)
                result = result[::scale, ::scale, :]
                net['output'][l].append(result)
    penultimate_layer = np.array([])
    for i in range(input_maps):
        cnt_map = net['output'][layer_num - 2][i].reshape(map_size[0] * map_size[1], map_size[2])
        penultimate_layer = np.vstack((penultimate_layer, cnt_map))
    final_output = net['weights'][layer_num - 1].dot(penultimate_layer)
    final_output -= np.max(final_output, axis=0).reshape(1, num_cases)
    final_output = np.exp(final_output)
    final_output /= np.sum(final_output, axis=0).reshape(1, num_cases)
    net['output'].append(final_output)
    net['cross_error'] = -(final_output * np.log(final_output)).sum() / num_cases


def cnn_backward(net, y_train):
    layer_num = net['layer_num']
    net_architecture = net['architecture']
    input_maps = net_architecture[0]['input_maps']
    output_maps = net_architecture[1]['output_maps']
    net['delta'] = []
    net['weights_gradient'] = []
    for l in range(layer_num - 1):
        net['delta'].append([])
        net['weights_gradient'].append([])
        for i in range(input_maps):
            net['weights_gradient'][l].append([])
            net['delta'][l].append(np.zeros(net['output'][l][i]))
            if net_architecture[l]['type'] == 'convolutional':
                for j in range(output_maps):
                    net['weights_gradient'][l][i].append(np.zeros((net['weights'][l][i][j].shape)))
        if net_architecture[l]['type'] == 'convolutional':
            input_maps = net_architecture[l]['output_maps']
    net['delta'].append(np.zeros(net['output'][layer_num - 1]))
    net['delta'][layer_num - 1] = (net['output'][layer_num - 1] - y_train)
    pre_layer = net['weights'].T.dot(net['delta'][layer_num - 1])
    map_size = net['output'][layer_num - 2][0].shape
    output_maps = net_architecture[layer_num - 3]['output_maps']
    for i in range(output_maps):
        map_num = map_size[0] * map_size[1]
        net['delta'][layer_num - 1][i] = pre_layer[i * map_num : (i + 1) * map_num, :].reshape(
            map_size[0], map_size[1], map_size[2])
    for l in range(layer_num - 3, 0, -1):
        if net_architecture[l]['type'] == 'pooling':
            input_maps = net_architecture[l - 1]['output_maps']
            for i in range(input_maps):
                result = np.zeros(net['output'][l][i].shape)
                for j in range(output_maps):
                    result += signal.convolve2d(net['delta'][l+1][j], np.rot90(net['weights'][l][i][j], 2), 'full')
                net['delta'][l][i] = result
        elif net_architecture[l]['type'] == 'convolutional':
            scale = net_architecture[l]['scale']
            for i in range(output_maps):
                net['delta'][l][i] = np.kron(net['delta'][l + 1][i], np.ones((scale, scale))) / (scale ** 2)
                net['delta'][l][i] *= net['output'][l][i]
    input_maps = net_architecture[0]['input_maps']
    for l in range(layer_num - 2):




def cnn_apply_grad():
    pass

def cnn_predict(net, x_test, y_test):
    pass


def cnn_train(net, x_train, y_train):
    num_cases = x_train.shape[0]
    for i in range(net['num_epoch']):
        tot_error_num = 0
        tot_cross_error = 0
        batch_size = net['batch_size']
        random_index = np.random.permutation(num_cases)
        print 'epoch %d' % i,
        for j in range(num_cases / batch_size):
            cnt_batch = x_train[random_index[j * batch_size: (j + 1) * batch_size], :]
            cnt_target = y_train[random_index[j * batch_size : (j + 1) * batch_size], :]
            cnn_forward(net, cnt_batch, cnt_target)
            cnn_backward(net, cnt_target)
            cnn_apply_grad(net)
            error_num = cnn_predict(net, cnt_batch, cnt_target)
            tot_error_num += error_num
            tot_cross_error += net['cross_error']
        print 'classification error :%f, cross_entropy error:%f' % (tot_error_num * 1.0 / num_cases, tot_cross_error)


def cnn_check_gradient():
    pass


def cnn_test():
    x_train, y_train, x_test, y_test = get_data()
    x_train = x_train.astype('float')
    x_test = x_test.astype('float')
    x_train = x_train.reshape((28, 28, 60000)) / 255
    x_test = x_test.reshape((28, 28, 10000)) / 255
    y_train = y_train.T
    y_test = y_test.T
    plt.figure(1, (12, 6))
    x = x_train[:,:, 0]
    plt.imshow(x, cmap=plt.gray())
    plt.show()
    net_architecture = [{'type': 'input', 'input_maps': 1},
                 {'type': 'convolutional', 'output_maps': 6, 'kernel_size': 5},
                 {'type': 'pooling', 'scale': 2},
                 {'type': 'convolutional', 'output_maps': 12, 'kernel_size': 5},
                 {'type': 'pooling', 'scale': 2}
                 ]
    net = cnn_setup(net_architecture, x_train, y_train)
    cnn_train(net, x_train, y_train)
    error_num = cnn_predict(net, x_test, y_test)
    print 'after trained %d misclassification' % error_num



if __name__ == '__main__':
    cnn_check_gradient()
    cnn_test()