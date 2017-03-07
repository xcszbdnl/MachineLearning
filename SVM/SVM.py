import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEPARABLE = False
GEN_W = 1
GEN_B = 0
LOW = -25
HIGH = 25
DATA_NUM = 50
MAX_RANGE = 5
TOLERANCE = 1e-5  # update tolerance for each variable
EPSILON = 1e-5
MAX_ITER = 100
C = 10


def generate_data():
    np.random.seed(42)
    # to generate data around y = GEN_W * x + GEN_B
    x_train = np.zeros((DATA_NUM, 2))
    y_train = np.zeros(DATA_NUM)
    if SEPARABLE is False:
        label_change_prob = 0.1
    else:
        label_change_prob = 0
    i = 0
    while i < DATA_NUM:
        x_1 = (HIGH - LOW) * np.random.random() + LOW
        x_2 = (HIGH - LOW) * np.random.random() + LOW
        label = 0
        if x_2 > GEN_W * x_1 + GEN_B:
            label = 1
        else:
            label = -1
        if abs(x_2 - GEN_W * x_1 - GEN_B) < MAX_RANGE:
            if SEPARABLE is True:
                i -= 1  # in order to separate more clearly
                continue
            label_change = np.random.rand()  # whether to convert the given label
            if label_change < label_change_prob:
                label *= -1
        x_train[i, 0] = x_1
        x_train[i, 1] = x_2
        y_train[i] = label
        i += 1
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.show()
    return x_train, y_train


def get_kernel(i, j, x_train):
    a = x_train[i, :]
    b = x_train[j, :]
    return a.dot(b)


def get_e(index, param_alpha, b, x_train, y_train):
    result = 0.0
    for i in range(DATA_NUM):
        result += param_alpha[i] * y_train[i] * get_kernel(index, i, x_train)
    result += b[0]
    result -= y_train[index]
    return result


def select_j(index, e_i, param_alpha, b, x_train, y_train, e_cache):
    j = 0
    e_j = get_e(j, param_alpha, b, x_train, y_train)
    max_delta = 0
    e_cache[index, :] = [1, e_i]
    valid_index = np.nonzero(e_cache[:, 0])[0]
    if len(valid_index) > 1:
        for k in valid_index:
            if k == index:
                continue
            e_k = get_e(k, param_alpha, b, x_train, y_train)
            if abs(e_i - e_k) > max_delta:
                max_delta = abs(e_i - e_k)
                e_j = e_k
                j = k
    else:
        j = index
        while j == index:
            j = int(np.random.uniform(0, DATA_NUM))
        e_j = get_e(j, param_alpha, b, x_train, y_train)
    return j, e_j


def clip_alpha(L, H, alpha_j_new):
    if alpha_j_new < L:
        alpha_j_new = L
    elif alpha_j_new > H:
        alpha_j_new = H
    return alpha_j_new


def update_e(i, param_alpha, b, x_train, y_train, e_cache):
    e_i = get_e(i, param_alpha, b, x_train, y_train)
    e_cache[i, :] = [1, e_i]


def random_select(i, e_i, param_alpha, b, x_train, y_train, e_cache):
    j = i
    e_j = 0
    while j == i:
        j = np.random.randint(0, DATA_NUM)
        e_j = get_e(j, param_alpha, b, x_train, y_train)
    return j, e_j


def update_alpha(i, param_alpha, b, x_train, y_train, e_cache):
    e_i = get_e(i, param_alpha, b, x_train, y_train)
    if (e_i * y_train[i] < -EPSILON and param_alpha[i] < C) or \
            (e_i * y_train[i] > EPSILON and param_alpha[i] > 0):
        j, e_j = select_j(i, e_i, param_alpha, b, x_train, y_train, e_cache)
        # j, e_j = random_select(i, e_i, param_alpha, b, x_train, y_train, e_cache)
        eta = get_kernel(i, i, x_train) + get_kernel(j, j, x_train) - 2 * get_kernel(i, j, x_train)
        if eta <= 0.0:
            print 'eta <= 0'
            return 0
        alpha_i_old = param_alpha[i].copy()
        alpha_j_old = param_alpha[j].copy()
        if y_train[i] == y_train[j]:
            H = min(C, alpha_i_old + alpha_j_old)
            L = max(0, alpha_i_old + alpha_j_old - C)
        else:
            H = min(C, C + alpha_j_old - alpha_i_old)
            L = max(0, alpha_j_old - alpha_i_old)
        if L == H:
            print 'L equals H'
            return 0
        alpha_j_new = alpha_j_old + y_train[j] * (e_i - e_j) / eta
        alpha_j_new = clip_alpha(L, H, alpha_j_new)
        update_e(j, param_alpha, b, x_train, y_train, e_cache)
        param_alpha[j] = alpha_j_new
        if abs(alpha_j_new - alpha_j_old) < TOLERANCE:
            print 'j not moving enough'
            return 0
        alpha_i_new = alpha_i_old + y_train[i] * y_train[j] * (alpha_j_old - alpha_j_new)
        param_alpha[i] = alpha_i_new
        update_e(i, param_alpha, b, x_train, y_train, e_cache)
        b_1 = -e_i - y_train[i] * get_kernel(i, i, x_train) * (alpha_i_new - alpha_i_old) - y_train[j] * \
            get_kernel(i, j, x_train) * (alpha_j_new - alpha_j_old) + b[0]
        b_2 = -e_j - y_train[i] * get_kernel(i, j, x_train) * (alpha_i_new - alpha_i_old) - y_train[j] * \
            get_kernel(j, j, x_train) * (alpha_j_new - alpha_j_old) + b[0]
        if (alpha_i_new > 0) and (alpha_i_new < C):
            b[0] = b_1
        elif (alpha_j_new > 0) and (alpha_j_new < C):
            b[0] = b_2
        else:
            b[0] = (b_1 + b_2) / 2
        return 1
    return 0


def draw_line(param_alpha, b, x_train, y_train):
    weight_w = np.zeros(2)
    for i in range(DATA_NUM):
        weight_w += param_alpha[i] * y_train[i] * x_train[i, :]
    k = -weight_w[0] / weight_w[1]
    b[0] /= -weight_w[1]
    r = 1 / weight_w[1]
    point_x = [x_train[:, 0].min(), x_train[:, 0].max()]
    point_y = []
    point_y_up = []
    point_y_down = []
    for x in point_x:
        y = x * k + b[0]
        y_1 = x * k + b[0] + r
        y_2 = x * k + b[0] - r
        point_y.append(y)
        point_y_up.append(y_1)
        point_y_down.append(y_2)
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.plot(point_x, point_y, 'b')
    plt.plot(point_x, point_y_up, 'b--')
    plt.plot(point_x, point_y_down, 'b--')
    plt.show()


def svm():
    x_train, y_train = generate_data()
    param_alpha = np.zeros(DATA_NUM)
    b = [0]
    entire_set = True
    iter_num = 0
    pair_changed = 0
    e_cache = np.zeros((DATA_NUM, 2))
    while iter_num < MAX_ITER and (entire_set is True or pair_changed > 0):
        pair_changed = 0
        if entire_set is True:
            for i in range(DATA_NUM):
                pair_changed += update_alpha(i, param_alpha, b, x_train, y_train, e_cache)
            print 'entire set, iter:%d, pair changed:%d' % (iter_num, pair_changed)
        else:
            unbounded_index = np.nonzero((param_alpha > 0) * (param_alpha < C))[0]
            for i in unbounded_index:
                pair_changed += update_alpha(i, param_alpha, b, x_train, y_train, e_cache)
                print 'unbounded alpha, iter:%d, index:%d, pair changed:%d' % (iter_num, i, pair_changed)
        iter_num += 1
        if entire_set is True:
            entire_set = False
        elif pair_changed == 0:
            entire_set = True
        print 'iter num: %d' % iter_num
    draw_line(param_alpha, b, x_train, y_train)


if __name__ == '__main__':
    svm()
