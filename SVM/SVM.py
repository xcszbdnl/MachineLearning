import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEPARABLE = False
GEN_W = 1
GEN_B = 0
LOW = -25
HIGH = 25
DATA_NUM = 100
MAX_RANGE = 10
TOLERANCE = 1e-5  # update tolerance for each variable


def generate_data():
    np.random.seed(40)
    # to generate data around y = GEN_W * x + GEN_B
    x_train = np.zeros((DATA_NUM, 2))
    y_train = np.zeros(DATA_NUM)
    if SEPARABLE is False:
        label_change_prob = 0.1
    else:
        label_change_prob = 0

    for i in range(DATA_NUM):
        x_1 = np.random.randint(LOW, HIGH)
        x_2 = np.random.randint(LOW, HIGH)
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
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.show()
    return x_train, y_train

def svm():

    x_train, y_train = generate_data()

if __name__ == '__main__':
    svm()