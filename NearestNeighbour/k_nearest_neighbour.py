import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbours = 15

def get_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


def standard_model():
    X, y = get_data()
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    n_neighbors = 15
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                 % (n_neighbors, weights))
    plt.show()


def dist(x_1, x_2):
    x = x_1[0] - x_2[0]
    y = x_1[1] - x_2[1]
    x *= x
    y *= y
    x += y
    return x


def predict_label(predict_data, x_train, y_train):

    num_cases = x_train.shape[0]
    neighbor_dist = np.zeros(num_cases)
    for i in range(num_cases):
        cnt_dist = dist(predict_data, x_train[i, :])
        neighbor_dist[i] = cnt_dist
    label_num = len(np.unique(y_train))
    votes = neighbor_dist.argsort()[: n_neighbours]
    labels = np.zeros(label_num)
    for i in range(n_neighbours):
        labels[y_train[votes[i]]] += 1
    return (-labels).argsort()[0]


def k_nearest_neighbour():
    x_train, y_train = get_data()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #step size for the mesh
    h = 0.02
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    predict_x = np.c_[xx.ravel(), yy.ravel()]
    predict_y = np.zeros((predict_x.shape[0]))
    predict_case = predict_x.shape[0]
    for i in range(predict_case):
        print 'case %d finished' % i
        predict_y[i] = predict_label(predict_x[i, :], x_train, y_train)
    predict_y = predict_y.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, predict_y, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('%d nearest neighbours decision boundary' % n_neighbours)
    plt.show()




if __name__ == '__main__':
    #standard_model()
    k_nearest_neighbour()