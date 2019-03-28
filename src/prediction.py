from keras.models import load_model
from keras.datasets import mnist
from utils import print_boundary
import numpy as np
import random
from matplotlib import pyplot as plt


def plot_image(image, label_true=None, class_names=None, label_pred=None):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    plt.grid()
    plt.imshow(image.astype(np.uint8))

    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: "+labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]

            xlabel = "True: "+labels_true_name+"\nPredicted: " + labels_pred_name

        # Show the class on the x-axis
        plt.xlabel(xlabel)

    plt.xticks([]) # Remove ticks from the plot
    plt.yticks([])
    plt.show() # Show the plot


# boundary_model = load_model("../model/boundary_mnist_cnn.h5")
class_model = load_model("../model/lenet5_label01_newprepocess.h5")

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# for i in range(100):
#     index = random.randint(0, 59999)
#     boundary_result = boundary_model.predict(x_train[index].reshape(1, 28, 28, 1)).argmax(axis=-1)[0]
#     class_result = class_model.predict(x_train[index].reshape(1, 28, 28, 1)).argmax(axis=-1)[0]
#     print_boundary(10, boundary_result)
#     print("class number: ", class_result)

discriminator = load_model("../model/new_prepocess_2label_boundary01_side1.h5")
# discriminator = load_model("../model/2label_boundary01_side1.h5")

data = np.load("../data/original_data/test_class_1.npz")
# data = np.load("../data/adversary_data/ori_class_0.npz")
# data = np.load("../data/adversary_data/target_1.npz")
# data = np.load("../data/2_label_model/data_0&1/new_prepocess_data_0&1_1000_part1.npz")
# data = np.load("../data/gan_training_each_boundary_data/boundary0_side0_data.npz")
# data = np.load("../data/adversary_data/new_prepocess_2label_ori_class_0.npz")
x_train = data["x_train"]
# x_train = x_train[1]


noise = np.random.normal(0, 1, (28, 28))
# print(noise)
# print(discriminator.predict(noise.reshape(1, 28, 28, 1)))
# print(x_train[0][0])
# img = x_train[0][0] * 255
# img = img.astype(int)
# plot_image(img.reshape(28, 28))
# print(x_train[0])
for i in range(100, 200):
    # print(class_model.predict(x_train[i].reshape(1, 28, 28, 1) / 127.5 - 1.).argmax(axis=-1)[0])
    # print(class_model.predict(x_train[i].reshape(1, 28, 28, 1) / 255).argmax(axis=-1)[0])
    # print(class_model.predict(x_train[i].reshape(1, 28, 28, 1)).argmax(axis=-1)[0])
    discriminator_result = discriminator.predict(x_train[i].reshape(1, 28, 28, 1) / 127.5 - 1.)
    # discriminator_result = discriminator.predict(x_train[i].reshape(1, 28, 28, 1) / 255)
    # discriminator_result = discriminator.predict(x_train[i].reshape(1, 28, 28, 1))

    print(discriminator_result[0][0])
# for i in range(100):
#     index = random.randint(0, 59999)
#     boundary_result = discriminator.predict(x_train[index].reshape(1, 28, 28, 1))[0][0]
#     print("near the boundary?: ", boundary_result)
# discriminator.summary()
