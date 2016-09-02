import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy
import operator
import os


def get_data():
    data_path = '../Data/mnist.npz'
    data_set = np.load(data_path)
    return data_set['x_train'], data_set['y_train'], data_set['x_test'], data_set['y_test']


def flip_all(x):
    dimension = len(x.shape)
    for i in range(dimension):
        x = x.swapaxes(i, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, i)
    return x


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
    net['num_epoch'] = 100
    net['batch_size'] = 100
    net['learning_rate'] = 0.1

    # the number of the layer do not include the fully connected layer
    layer_num = len(net_architecture)
    input_maps = net_architecture[0]['input_maps']
    map_size = ([x_train.shape[0], x_train.shape[1]])
    net['layer_num'] = layer_num
    # let the layer of bias index start at 1, weights index start at 0
    net['bias'].append([])
    for l in range(1, layer_num):
        net['weights'].append([])
        net['bias'].append([])
        if net_architecture[l]['type'] == 'convolutional':
            kernel_size = net_architecture[l]['kernel_size']
            map_size = tuple(x - kernel_size + 1 for x in map_size)
            fan_out = net_architecture[l]['output_maps'] * (kernel_size ** 2)
            for i in range(input_maps):
                net['weights'][l - 1].append([])
                fan_in = input_maps * (kernel_size ** 2)
                for j in range(net_architecture[l]['output_maps']):
                    net['weights'][l - 1][i].append(((np.random.rand(kernel_size, kernel_size) - 0.5) *
                                                 2 * np.sqrt(6.0 / (fan_in + fan_out))))
            for i in range(net_architecture[l]['output_maps']):
                net['bias'][l].append(0)
            input_maps = net_architecture[l]['output_maps']

        elif net_architecture[l]['type'] == 'pooling':
            map_size = tuple(x / net_architecture[l]['scale'] for x in map_size)

    final_num = np.prod(map_size) * input_maps
    output_num = y_train.shape[0]
    net['weights'].append((np.random.rand(output_num, final_num) - 0.5) * 2 * np.sqrt(6.0 / (final_num + output_num)))
    return net


def cnn_forward(net, x_train, y_train):
    net_architecture = net['architecture']
    net['output'] = []
    net['output'].append([])
    net['output'][0].append(x_train)
    num_cases = x_train.shape[2]
    layer_num = net['layer_num']
    input_maps = net_architecture[0]['input_maps']
    for l in range(1, layer_num):
        net['output'].append([])
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
            kernel_size = net_architecture[l]['kernel_size']
            for i in range(output_maps):
                map_size = tuple(map(operator.sub, net['output'][l - 1][0].shape, (kernel_size - 1, kernel_size - 1, 0)))
                result = np.zeros(map_size)
                for j in range(input_maps):
                    result += signal.convolve(net['output'][l - 1][j], net['weights'][l - 1][j][i][..., None], 'valid')
                result += net['bias'][l][i]
                vector_sigmoid = np.vectorize(sigmoid)
                result = vector_sigmoid(result)
                net['output'][l].append(result)
            input_maps = output_maps
        elif net_architecture[l]['type'] == 'pooling':
            scale = net_architecture[l]['scale']
            mean_pooling = np.ones((scale, scale)) / (scale ** 2)
            for i in range(output_maps):
                result = signal.convolve(net['output'][l - 1][i], mean_pooling[..., None], 'valid')
                result = result[::scale, ::scale, :]
                net['output'][l].append(result)
    penultimate_layer = np.array([])
    for i in range(input_maps):
        map_size = net['output'][layer_num - 1][i].shape
        cnt_map = net['output'][layer_num - 1][i].reshape(map_size[0] * map_size[1], map_size[2])
        penultimate_layer = np.vstack((penultimate_layer, cnt_map)) if penultimate_layer.size else cnt_map
    net['penultimate_layer'] = penultimate_layer
    final_output = net['weights'][layer_num - 1].dot(penultimate_layer)
    final_output -= np.max(final_output, axis=0).reshape(1, num_cases)
    final_output = np.exp(final_output)
    final_output /= np.sum(final_output, axis=0).reshape(1, num_cases)
    net['output'].append(final_output)
    net['cross_error'] = -(y_train * np.log(final_output)).sum() / num_cases


def cnn_backward(net, y_train):
    layer_num = net['layer_num']
    net_architecture = net['architecture']
    input_maps = net_architecture[0]['input_maps']
    output_maps = input_maps
    num_cases = y_train.shape[1]
    net['delta'] = []
    net['weights_gradient'] = []
    net['bias_gradient'] = []
    net['delta'].append([])
    for i in range(input_maps):
        net['delta'][0].append(np.zeros(net['output'][0][i].shape))
    net['bias_gradient'].append([])
    for l in range(1, layer_num):
        net['delta'].append([])
        net['weights_gradient'].append([])
        net['bias_gradient'].append([])
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
        for i in range(output_maps):
            net['delta'][l].append(np.zeros(net['output'][l][i].shape))
            net['bias_gradient'][l].append(0)
        for i in range(input_maps):
            net['weights_gradient'][l - 1].append([])
            if net_architecture[l]['type'] == 'convolutional':
                for j in range(output_maps):
                    net['weights_gradient'][l - 1][i].append(np.zeros(net['weights'][l - 1][i][j].shape))
        if net_architecture[l]['type'] == 'convolutional':
            input_maps = net_architecture[l]['output_maps']
    net['weights_gradient'].append(np.zeros(net['weights'][layer_num - 1].shape))
    net['delta'].append(np.zeros(net['output'][layer_num].shape))
    net['delta'][layer_num] = (net['output'][layer_num] - y_train)
    pre_layer = net['weights'][layer_num - 1].T.dot(net['delta'][layer_num])
    map_size = net['output'][layer_num - 1][0].shape

    # convert fully connected layer to image layer
    for i in range(output_maps):
        map_num = map_size[0] * map_size[1]
        net['delta'][layer_num - 1][i] = pre_layer[i * map_num : (i + 1) * map_num, :].reshape(
            map_size[0], map_size[1], map_size[2])

    for l in range(layer_num - 2, 0, -1):
        if net_architecture[l]['type'] == 'pooling':
            input_maps = net_architecture[l - 1]['output_maps']
            for i in range(input_maps):
                result = np.zeros(net['output'][l][i].shape)
                for j in range(output_maps):
                    weights = net['weights'][l][i][j]
                    weights = np.rot90(weights, 2)
                    weights = weights[..., None]
                    result += signal.convolve(net['delta'][l + 1][j], weights, 'full')
                net['delta'][l][i] = result
            output_maps = input_maps
        elif net_architecture[l]['type'] == 'convolutional':
            scale = net_architecture[l + 1]['scale']
            for i in range(output_maps):
                sigmoid_gradient = net['output'][l][i] * (1 - net['output'][l][i])
                net['delta'][l][i] = np.kron(net['delta'][l + 1][i], np.ones((scale, scale))[..., None]) / (scale ** 2)
                net['delta'][l][i] *= sigmoid_gradient

    input_maps = net_architecture[0]['input_maps']
    for l in range(layer_num):
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
            for i in range(output_maps):
                # calculate gradient for different filters
                for j in range(input_maps):
                    output_flip = flip_all(net['output'][l - 1][j])
                    gradient_result = signal.convolve(output_flip, net['delta'][l][i], 'valid')
                    gradient_result = gradient_result.reshape(gradient_result.shape[0], gradient_result.shape[1])
                    gradient_result /= num_cases
                    net['weights_gradient'][l - 1][j][i] = gradient_result
                # calculate bias
                net['bias_gradient'][l][i] = np.sum(net['delta'][l][i]) / num_cases
    net['weights_gradient'][layer_num - 1] = net['delta'][layer_num].dot(net['penultimate_layer'].T) / num_cases


def cnn_apply_grad(net):
    layer_num = net['layer_num']
    net_architecture = net['architecture']
    input_maps = net_architecture[0]['input_maps']
    learning_rate = net['learning_rate']
    for l in range(layer_num):
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
            for i in range(output_maps):
                net['bias'][l][i] = net['bias'][l][i] - learning_rate * net['bias_gradient'][l][i]
                for j in range(input_maps):
                    net['weights'][l - 1][j][i] = net['weights'][l - 1][j][i] - learning_rate * net['weights_gradient'][l - 1][j][i]
    net['weights'][layer_num - 1] = net['weights'][layer_num - 1] - learning_rate * net['weights_gradient'][layer_num - 1]


def cnn_predict(net, x_test, y_test):
    cnn_forward(net, x_test, y_test)
    predict = net['output'][net['layer_num']]
    label = np.argsort(-predict, axis=0)[0, :]
    err_num = 0
    for i in range(y_test.shape[1]):
        if y_test[label[i], i] == 0:
            err_num += 1
    return err_num


def cnn_train(net, x_train, y_train):
    num_cases = x_train.shape[2]
    for i in range(net['num_epoch']):
        tot_error_num = 0
        tot_cross_error = 0
        batch_size = net['batch_size']
        random_index = np.random.permutation(num_cases)
        print 'epoch %d' % i,
        for j in range(num_cases / batch_size):
            cnt_batch = x_train[:, :, random_index[j * batch_size: (j + 1) * batch_size]]
            cnt_target = y_train[:, random_index[j * batch_size : (j + 1) * batch_size]]
            cnn_forward(net, cnt_batch, cnt_target)
            cnn_backward(net, cnt_target)
            cnn_apply_grad(net)
            error_num = cnn_predict(net, cnt_batch, cnt_target)
            tot_error_num += error_num
            tot_cross_error += net['cross_error']
        print 'classification error :%f, cross_entropy error:%f' % (tot_error_num * 1.0 / num_cases, tot_cross_error)


def gradient_numerical_check(net, x_train, y_train):
    epsilon = 1e-4
    error_threshold = 1e-7
    layer_num = net['layer_num']
    net_architecture = net['architecture']
    flag_err = True
    for i in range(net['weights'][layer_num - 1].shape[0]):
        for j in range(net['weights'][layer_num - 1].shape[1]):
            net_p = copy.deepcopy(net)
            net_n = copy.deepcopy(net)
            net_p['weights'][layer_num - 1][i][j] += epsilon
            net_n['weights'][layer_num - 1][i][j] -= epsilon
            cnn_forward(net_p, x_train, y_train)
            cnn_forward(net_n, x_train, y_train)
            delta = (net_p['cross_error'] - net_n['cross_error']) / (2 * epsilon)
            err = np.abs(delta - net['weights_gradient'][layer_num - 1][i][j])
            if err > error_threshold:
                flag_err = False
                print 'gradient calculate error at layer %d, index: (%d, %d)' % (layer_num, i, j)

    input_maps = net_architecture[0]['input_maps']
    for l in range(layer_num):
        if net_architecture[l]['type'] == 'convolutional':
            output_maps = net_architecture[l]['output_maps']
            for i in range(output_maps):
                net_p = copy.deepcopy(net)
                net_n = copy.deepcopy(net)
                net_p['bias'][l][i] += epsilon
                net_n['bias'][l][i] -= epsilon
                cnn_forward(net_p, x_train, y_train)
                cnn_forward(net_n, x_train, y_train)
                delta = (net_p['cross_error'] - net_n['cross_error']) / (2 * epsilon)
                err = np.abs(delta - net['bias_gradient'][l][i])
                if err > error_threshold:
                    flag_err = False
                    print 'gradient calculate error at layer %d, bias index: %d' % (l, i)

                for j in range(input_maps):
                    for u in range(net['weights'][l - 1][j][i].shape[0]):
                        for v in range(net['weights'][l - 1][j][i].shape[1]):
                            net_p = copy.deepcopy(net)
                            net_n = copy.deepcopy(net)
                            net_p['weights'][l - 1][j][i][u][v] += epsilon
                            net_n['weights'][l - 1][j][i][u][v] -= epsilon
                            cnn_forward(net_p, x_train, y_train)
                            cnn_forward(net_n, x_train, y_train)
                            delta = (net_p['cross_error'] - net_n['cross_error']) / (2 * epsilon)
                            err = np.abs(delta - net['weights_gradient'][l - 1][j][i][u][v])
                            if err > error_threshold:
                                flag_err = False
                                print 'gradient calculate error at layer %d, input_map: %d,' \
                                      ' output_map : %d, index: (%d, %d)' % (l, j, i, u, v)
    if flag_err:
        print 'no error found, correct bp algorithm'


def cnn_check_gradient():
    x_train = np.random.rand(28, 28, 10)
    y_train = np.random.rand(10, 10)
    y_train = y_train == np.repeat(np.max(y_train, axis=0).reshape(1, y_train.shape[1]), y_train.shape[0], axis=0)
    y_train = y_train.astype('int')

    net_architecture = [{'type': 'input', 'input_maps': 1},
                 {'type': 'convolutional', 'output_maps': 6, 'kernel_size': 5},
                 {'type': 'pooling', 'scale': 2},
                 {'type': 'convolutional', 'output_maps': 12, 'kernel_size': 5},
                 {'type': 'pooling', 'scale': 2},
                 ]
    """
    net_architecture = [{'type': 'input', 'input_maps': 1}
                 ]
    """
    net = cnn_setup(net_architecture, x_train, y_train)
    cnn_forward(net, x_train, y_train)
    cnn_backward(net, y_train)
    gradient_numerical_check(net, x_train, y_train)


def cnn_test():
    # all training vector are column vector
    # training matrix should be reshaped like (image width * image height * dataset size)
    # label matrix should be reshaped like (label number * dataset size)
    x_train, y_train, x_test, y_test = get_data()
    x_train = x_train.astype('float')
    x_test = x_test.astype('float')
    x_train = x_train.T.reshape((28, 28, 60000)) / 255
    x_test = x_test.T.reshape((28, 28, 10000)) / 255
    y_train = y_train.T
    y_test = y_test.T
    """
    plt.figure(1, (12, 6))
    x = x_train[:, :, 0]
    plt.imshow(x, cmap=plt.gray())
    plt.show()
    """
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
    np.random.seed(42)
    # cnn_check_gradient()
    cnn_test()