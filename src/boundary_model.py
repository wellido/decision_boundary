"""
code from keras.io https://keras.io/examples/mnist_cnn/
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from sklearn.utils import shuffle

batch_size = 128
num_classes = 55
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets

print("original training data...")
ori_train_len = []
x_train = np.load("../data/original_data/class_0.npz")["x_train"] / 255
ori_train_len.append(len(x_train))
for i in range(1, 10):
    x_s = np.load("../data/original_data/class_" + str(i) + ".npz")["x_train"] / 255
    x_train = np.append(x_train, x_s, axis=0)
    ori_train_len.append(len(x_s))
print("completed")

print("original test data...")
ori_test_len = []
x_test = np.load("../data/original_data/test_class_0.npz")["x_train"] / 255
ori_test_len.append(len(x_test))
for i in range(1, 10):
    x_s = np.load("../data/original_data/test_class_" + str(i) + ".npz")["x_train"] / 255
    x_test = np.append(x_test, x_s, axis=0)
    ori_test_len.append(len(x_s))
print("completed")

print("boundary data...")
for i in range(9):
    for j in range(i + 1, 10):
        x_s = np.load("../data/float_boundary_data/data_" + str(i) + "&" + str(j) +
                      "/data_" + str(i) + "&" + str(j) + "_1000_part1.npz")["x_train"]
        x_train = np.append(x_train, x_s[0][:6000].reshape(6000, img_rows, img_cols), axis=0)
        x_test = np.append(x_test, x_s[0][6000:].reshape(1000, img_rows, img_cols), axis=0)
print("completed")

print("label...")
y_train = []
y_test = []
for i in range(10):
    y_train = y_train + [i for j in range(ori_train_len[i])]
    y_test = y_test + [i for j in range(ori_test_len[i])]

for i in range(10, 55):
    y_train = y_train + [i for j in range(6000)]
    y_test = y_test + [i for j in range(1000)]

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("completed")

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)

# mnist cnn
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

# lenet-5
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=input_shape, activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(120, activation='tanh'))
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(num_classes, activation='softmax'))
# sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, shuffle=True)
#
# score = model.evaluate(x_test, y_test, verbose=0)
model.save("../model/55label_test2.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
