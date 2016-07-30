import numpy as np
import matplotlib.pyplot as plt


def lda():
    n_sample = 100
    plt.figure(1, (12, 6))

    np.random.seed(42)

    gaussian_1 = np.random.randn(n_sample, 2) + np.array([10, 10])

    gaussian_2 = np.random.randn(n_sample, 2) + np.array([8, 8])


    plt.scatter(gaussian_1[:,0], gaussian_1[:,1], color='r')
    plt.scatter(gaussian_2[:,0], gaussian_2[:,1], color='b')

    plt.show()

    x_train = np.vstack((gaussian_1, gaussian_2))

    y_train = np.concatenate((np.ones(n_sample), np.zeros( n_sample)))

    class_1 = x_train[y_train == 1, :]
    class_2 = x_train[y_train == 0, :]

    mean_1 = class_1.mean(axis=0).reshape((2,1))
    mean_2 = class_2.mean(axis=0).reshape((2,1))

    class_1 = class_1.T - mean_1
    class_2 = class_2.T - mean_2

    s_w = class_1.dot(class_1.T) + class_2.dot(class_2.T)

    w = np.linalg.inv(s_w).dot(mean_2 - mean_1)

    w = w / np.sqrt(np.sum(np.abs(w) ** 2))

    k = w[1] / w[0]

    class_1 = class_1 + mean_1
    class_2 = class_2 + mean_2
    project_class_1 = np.zeros(class_1.shape)
    project_class_1[0, :] = k * class_1[1, :] + class_1[0, :] / (1 + k * k)
    project_class_1[1, :] = k * project_class_1[0, :]

    project_class_2 = np.zeros(class_2.shape)
    project_class_2[0, :] = k * class_2[1, :] + class_2[0, :] / (1 + k * k)
    project_class_2[1, :] = k * project_class_2[0, :]

    plt.figure(2, (12, 6))

    plt.scatter(project_class_1[0, :], project_class_1[1, :], color='r')
    plt.scatter(project_class_2[0, :], project_class_2[1, :], color='b')

    plt.show()

if __name__ == '__main__':
    lda()
