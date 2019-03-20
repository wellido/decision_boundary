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




