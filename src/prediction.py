from keras.models import load_model
from keras.datasets import mnist
from utils import print_boundary
import numpy as np
import random

# boundary_model = load_model("../model/boundary_mnist_cnn.h5")
# class_model = load_model("../model/lenet-5.h5")

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# for i in range(100):
#     index = random.randint(0, 59999)
#     boundary_result = boundary_model.predict(x_train[index].reshape(1, 28, 28, 1)).argmax(axis=-1)[0]
#     class_result = class_model.predict(x_train[index].reshape(1, 28, 28, 1)).argmax(axis=-1)[0]
#     print_boundary(10, boundary_result)
#     print("class number: ", class_result)

discriminator = load_model("../model/boundary0.h5")
# data = np.load("../data/original_data/class_0.npz")
data = np.load("../data/adversary_data/target_0.npz")
x_train = data["x_train"]

for i in range(100):

    discriminator_result = discriminator.predict(x_train[i].reshape(1, 28, 28, 1))
    print(discriminator_result)
# for i in range(100):
#     index = random.randint(0, 59999)
#     boundary_result = discriminator.predict(x_train[index].reshape(1, 28, 28, 1))[0][0]
#     print("near the boundary?: ", boundary_result)
# discriminator.summary()
