from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras import backend as K
from sklearn.utils import shuffle


def rgb2gray(rgb):
    return np.dot(rgb[..., : 3], [0.299, 0.587, 0.114])


# input image dimensions
img_rows, img_cols = 32, 32
num_classes = 11
batch_size = 128
epochs = 20

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

subtract_pixel_mean = True
# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

gray_train = rgb2gray(x_train)
gray_test = rgb2gray(x_test)

print("boundary data...")
for i in range(9):
    for j in range(i + 1, 10):
        x_s = np.load("../data/cifar10_boundary_data/data_" + str(i) + "&" + str(j) +
                      "/data_" + str(i) + "&" + str(j) + "_2000_part1.npz")["x_train"]
        # print(x_s.shape)
        # x_train = np.append(x_train, x_s[0][:6000].reshape(6000, img_rows, img_cols), axis=0)
        # x_test = np.append(x_test, x_s[0][6000:].reshape(1000, img_rows, img_cols), axis=0)
        x_add_train = rgb2gray(x_s[0][:111]).reshape(111, 32, 32)
        x_add_test = rgb2gray(x_s[0][-22:]).reshape(22, 32, 32)
        gray_train = np.append(gray_train, x_add_train, axis=0)
        gray_test = np.append(gray_test, x_add_test, axis=0)
print("completed")
y_train_boundary = [10 for i in range(4995)]
y_test_boundary = [10 for j in range(990)]
y_train_boundary = np.asarray(y_train_boundary)
y_test_boundary = np.asarray(y_test_boundary)
print(y_train_boundary.shape)
print(y_train.shape)
y_train = np.append(y_train, y_train_boundary.reshape(4995, 1))
y_test = np.append(y_test, y_test_boundary.reshape(990, 1))


if K.image_data_format() == 'channels_first':
    x_train = gray_train.reshape(gray_train.shape[0], 1, img_rows, img_cols)
    x_test = gray_test.reshape(gray_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = gray_train.reshape(gray_train.shape[0], img_rows, img_cols, 1)
    x_test = gray_test.reshape(gray_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# plt.imshow(x_train[0])
# plt.axis('off')
# plt.show()

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


# lenet-5
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=input_shape, activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(120, activation='tanh'))
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(num_classes, activation='softmax'))
# sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, shuffle=True)

# sadl mnist
# layers = [
#             Conv2D(64, (3, 3), padding="valid", input_shape=(32, 32, 1)),
#             Activation("relu"),
#             Conv2D(64, (3, 3)),
#             Activation("relu"),
#             MaxPooling2D(pool_size=(2, 2)),
#             Dropout(0.5),
#             Flatten(),
#             Dense(128),
#             Activation("relu"),
#             Dropout(0.5),
#             Dense(10),
#         ]
# model = Sequential()
# for layer in layers:
#     model.add(layer)
# model.add(Activation("softmax"))
# model.compile(
#         loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
#     )
# model.fit(
#         x_train,
#         y_train,
#         epochs=50,
#         batch_size=128,
#         shuffle=True,
#         verbose=1,
#     )

score = model.evaluate(x_test, y_test, verbose=0)
model.save("../model/cifar10_rgb2gray_test.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
