import numpy as np
from sklearn.utils import shuffle


def load_boundary_data(dir_path, label_num):
    """

    :param dir_path:
    :param label_num:
    :return:
    """
    x_train = []
    y_train = []
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    label_count = 0

    for i in range(label_num - 1):
        for j in range(i + 1, label_num):
            data_path = dir_path + "data_" + str(i) + "&" + str(j) + \
                        "/data_" + str(i) + "&" + str(j) + "_1000_part1.npz"
            single_data = np.load(data_path)
            single_x = single_data["x_train"]
            x_train = np.append(x_train, single_x[0]).reshape((2 * label_count + 1) * len(single_x[0]), 1, 28, 28, 1)
            x_train = np.append(x_train, single_x[1]).reshape((2 * label_count + 2) * len(single_x[0]), 1, 28, 28, 1)
            print(x_train.shape)
            y_tr = [label_count for k in range(int(len(single_x[0]) * 2))]
            y_tr = np.asarray(y_tr)
            y_train = np.append(y_train, y_tr)
            label_count += 1
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    return x_train, y_train


def load_single_label_boundary_data(dir_path, boundary_1, boundary_2, label_count):
    """

    :param dir_path:
    :param boundary_1:
    :param boundary_2:
    :param label_count:
    :return:
    """
    x_train = []
    x_train = np.asarray(x_train)
    data_path = dir_path + "data_" + str(boundary_1) + "&" + str(boundary_2) + \
                "/data_" + str(boundary_1) + "&" + str(boundary_2) + "_1000_part1.npz"
    single_data = np.load(data_path)
    single_x = single_data["x_train"]
    x_train = np.append(x_train, single_x[0]).reshape(len(single_x[0]), 1, 28, 28, 1)
    x_train = np.append(x_train, single_x[1]).reshape(2 * len(single_x[0]), 1, 28, 28, 1)
    y_tr = [label_count for k in range(int(len(single_x[0]) * 2))]
    y_tr = np.asarray(y_tr)
    return x_train, y_tr


def print_boundary(class_num, label):
    """

    :param class_num:
    :param label:
    :return:
    """
    label2boundary = []
    for i in range(class_num - 1):
        for j in range(i + 1, class_num):
            # print("label: %d, boundary: %d%d" % (count, i, j))
            label2boundary.append(str(i) + str(j))
    print("boundary: ", label2boundary[label])

