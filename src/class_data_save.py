from keras.datasets import mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_list = [[] for i in range(10)]
for i in range(60000):
    data_list[y_train[i]].append(x_train[i])
for i in range(10):
    np.savez("../data/class_" + str(i) + ".npz", x_train=data_list[i])
