import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 50, num=100)
y = x + np.random.randn(1, 100)
x_train = np.vstack((x,y))
x_train = x_train.T

plt.figure(1, (12,6))

plt.scatter(x_train[:,0], x_train[:,1])

plt.show()


x_mean = x_train.mean(axis=0)

x_train = x_train - x_mean

S = np.dot(x_train.T, x_train)

eigen_value, eigen_vector = np.linalg.eig(S)

max_index = np.argsort(-eigen_value)

component_num = 1

main_component_index = max_index[0:component_num]

max_component = eigen_vector[:, main_component_index]

main_component = max_component.T.dot(x_train.T)

plt.figure(2, (12, 6))

plt.scatter(main_component, np.zeros(main_component.shape[1]))

plt.title('main component after pca')

plt.show()

x_reconstruction = max_component.dot(main_component) + x_mean.reshape((1,2)).T

plt.figure(3, (12, 6))

plt.scatter(x_reconstruction[0,:], x_reconstruction[1,:])

plt.show()

print x_train.shape

