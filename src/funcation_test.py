from utils import load_boundary_data
import numpy as np

x_train, y_train = load_boundary_data("../data/boundary_data/", 10)
np.savez("../data/gan_training_data/gan_training_data.npz", x_train=x_train, y_train=y_train)
print("save completed.")