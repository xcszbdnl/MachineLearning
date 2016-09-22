import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

def get_data():
    diabetes = datasets.load_diabetes()
    x = diabetes.data
    y = diabetes.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    return x_train, y_train, x_test, y_test



def elastic_net():
    x_train, y_train, x_test, y_test = get_data()

if __name__ == '__main__':
    np.random.seed(42)
    elastic_net()