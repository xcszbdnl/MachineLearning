import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification

CLASSIFIER_NUM = 5
STEPS = 10


def get_data():
    n_samples = 100
    x, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, random_state=1,
                               n_clusters_per_class=1)
    y = y - (y == 0)
    rng = np.random.RandomState(2)
    x += 2 * rng.uniform(size=x.shape)
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
    plt.show()
    return x, y


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
        predict_label = (x_train[:, self.dim] > self.threshold).astype('int') * self.flag
        predict_label = predict_label - (predict_label == 0) * self.flag
        return predict_label


def train_classifier(dataset, init_weight):
    min_error = np.Inf
    best_classifier = WeakClassifier()
    dimensions = dataset['x_train'].shape[1]
    for i in range(dimensions):
        dimension_max = dataset['x_train'][:, i].max()
        dimension_min = dataset['x_train'][:, i].min()
        step_size = (dimension_max - dimension_min) / STEPS
        for j in range(-1, STEPS + 2):
            cnt_threshold = dimension_min + (j * step_size)
            for flag in [-1, 1]:
                cnt_classifier = WeakClassifier()
                cnt_classifier.dim = i
                cnt_classifier.threshold = cnt_threshold
                cnt_classifier.flag = flag
                predict_label = cnt_classifier.predict(dataset['x_train'])
                error = np.sum(init_weight * (predict_label != dataset['y_train']))
                weight_sum = np.sum(init_weight)
                epsilon_m = error / weight_sum
                epsilon_m = np.clip(epsilon_m, 1e-7, 1 - 1e-7)
                cnt_classifier.alpha = np.log((1 - epsilon_m) / epsilon_m)
                if error < min_error:
                    min_error = error
                    best_classifier = cnt_classifier
    return best_classifier


def update_weight(dataset, classifier, weight):
    exp_2_alpha = np.exp(-classifier.alpha / 2.0)
    predict_label = classifier.predict(dataset['x_train'])
    exp_alpha = np.exp(classifier.alpha * (predict_label != dataset['y_train']))
    weight *= exp_2_alpha * exp_alpha
    return weight


def adaboost_predict(data, classifiers):
    result = np.zeros(data.shape[0])
    for classifier in classifiers:
        result += classifier.alpha * classifier.predict(data)
    return 1 - 2 * result < 0


def decision_boundary(dataset, classifiers):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    x_max, x_min = dataset['x_train'][:, 0].max() + 1, dataset['x_train'][:, 0].min() - 1
    y_max, y_min = dataset['x_train'][:, 1].max() + 1, dataset['x_train'][:, 1].min() - 1
    h = 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    predict_x = np.c_[xx.ravel(), yy.ravel()]
    # predict_y = np.zeros(len(predict_x))
    # for i in range(len(predict_x)):
    #     predict_y[i] = adaboost_predict(predict_x[i, :], classifiers)
    predict_y = adaboost_predict(predict_x, classifiers)
    predict_y = predict_y.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, predict_y, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(dataset['x_train'][:, 0], dataset['x_train'][:, 1], c=dataset['y_train'], cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('adaboost decision boundary')
    plt.show()


def adaboost_train():
    x_train, y_train = get_data()
    dataset = dict(
        x_train=x_train,
        y_train=y_train,
    )
    num_cases = x_train.shape[0]
    weight = np.ones(num_cases, dtype='float32') / num_cases
    classifiers = []
    for _ in range(CLASSIFIER_NUM):
        cnt_classifier = train_classifier(dataset, weight)
        weight = update_weight(dataset, cnt_classifier, weight)
        classifiers.append(cnt_classifier)
    decision_boundary(dataset, classifiers)


if __name__ == '__main__':
    adaboost_train()
