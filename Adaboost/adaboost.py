import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

CLASSIFIER_NUM = 5
STEPS = 10


class WeakClassifier:
    def __init__(self):
        # the weight of the classifier
        self.alpha = 0.0
        # which dim to classify the data
        self.dim = -1
        # the threshold to classify the data
        self.threshold = .0
        # if feature larger than the threshold, then label = flag
        self.flag = -1

    def predict(self, x_train):
        predict_label = int(x_train[:, dim] > self.threshold) * self.flag
        return predict_label


def get_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


def train_classifier(dataset, init_weight):
    min_error = np.Inf


def adaboost_train():
    x_train , y_train = get_data()
    dataset = dict(
        x_train=x_train,
        y_train=y_train,
    )
    num_cases = x_train.shape[0]
    init_weight = np.ones((num_cases, 1), dtype='float32') / num_cases
    classifiers = []
    for _ in range(CLASSIFIER_NUM):
        cnt_classifier = train_classifier(dataset, init_weight)


if __name__ == '__main__':
    adaboost_train()

