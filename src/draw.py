# from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D
#
# mnist = datasets.load_digits()
# X = mnist.data
# y = mnist.target
# pca = decomposition.PCA(n_components=3)
# new_X = pca.fit_transform(X)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y)
# plt.show()


import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA, IncrementalPCA
# import seaborn as sns

data_first_list = np.load("../data/original_data/class_0.npz")["x_train"]
data_second_list = np.load("../data/original_data/class_1.npz")["x_train"]
boundary_data = np.load("../data/2_label_model/data_0&1/dichotomize_01_100000.npz")["x_train"]
boundary_0 = boundary_data[0][:500]
boundary_1 = boundary_data[1][:500]
boundary_0 = (boundary_0 + 1.) * 127.5
boundary_1 = (boundary_1 + 1.) * 127.5
boundary_0 = boundary_0.reshape(500, 784)
boundary_1 = boundary_1.reshape(500, 784)

data_first_list = data_first_list[:500].reshape(500, 784)
data_second_list = data_second_list[:500].reshape(500, 784)
data_draw = np.append(data_first_list, data_second_list, axis=0)
data_draw = np.append(data_draw, boundary_1, axis=0)
data_draw = np.append(data_draw, boundary_0, axis=0)
label = [0 for i in range(500)] + [1 for j in range(500)] + [2 for k in range(500)] \
        + [3 for m in range(500)]
label = np.asarray(label)
# print(len(label))

X = data_draw
y = label

pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(data_draw)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y)
plt.show()

# tsne
# standardized_data = StandardScaler().fit_transform(data_draw)
# print(standardized_data.shape)
# model = TSNE(n_components=3, random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
# reduced_data = model.fit_transform(standardized_data)
# reduced_df = np.vstack((reduced_data.T, label)).T
# reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y", "label"])
# reduced_df.label = reduced_df.label.astype(np.int)
#
# print(reduced_df.head())
# g = sns.FacetGrid(reduced_df, hue='label', size=6).map(plt.scatter, 'X', 'Y').add_legend()
# plt.show()

# pca

# estimator = PCA(n_components=2)
# x_pca = estimator.fit_transform(data_draw)
#
#
# def plot_pca_scatter():
#         # 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray'
#         colors = ['red', 'blue', 'purple', 'cyan']  # 总共有0-9,10种手写数字，要把每一个手写数字都用二维图展现出来，为了便于区分，使用10颜色
#         for i in range(len(colors)):
#                 px = x_pca[:, 0][label == i]  # 这里的[y_digits.as_matrix]主要对x_pca第一列的所有行起到选择作用，也就是说假设i=0时，
#                 py = x_pca[:, 1][label == i]  # 选择出所有训练样本的标签为0的x_pca，并将其用二维图展现出来，不同的数字用不同的颜色画出来
#                 plt.scatter(px, py, c=colors[i])  # 最后，通过最终效果图可以发现，同一类型的digits基本上分布在同一块区域
#
#         plt.legend(np.arange(0, 4).astype(str))
#         plt.xlabel('First Principal Component')
#         plt.ylabel('Sencond Principal Component')
#         plt.show()
# plot_pca_scatter()
