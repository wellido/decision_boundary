from utils import load_boundary_data
from utils import load_single_label_boundary_data
import numpy as np

# save all data
# x_train, y_train = load_boundary_data("../data/boundary_data/", 10)
# np.savez("../data/gan_training_data/gan_training_data.npz", x_train=x_train, y_train=y_train)
# print("save completed.")

# save one label data
x_train, y_train = load_single_label_boundary_data("../data/boundary_data/", 0, 1, 0)
np.savez("../data/gan_training_each_boundary_data/boundary01_data.npz", x_train=x_train, y_train=y_train)
print("save completed.")

