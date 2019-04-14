from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from sklearn.utils import shuffle

batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


# the data, split between train and test sets
print("original training data...")
x_train = np.load("../data/original_data/class_0.npz")["x_train"][:1000] / 255
for i in range(1, 10):
    x_s = np.load("../data/original_data/class_" + str(i) + ".npz")["x_train"][:1000] / 255
    x_train = np.append(x_train, x_s, axis=0)
print("completed")

print("original test data...")
x_test = np.load("../data/original_data/test_class_0.npz")["x_train"][:200] / 255
for i in range(1, 10):
    x_s = np.load("../data/original_data/test_class_" + str(i) + ".npz")["x_train"][:200] / 255
    x_test = np.append(x_test, x_s, axis=0)
print("completed")

print(len(x_train))
print("boundary data...")
for i in range(9):
    for j in range(i + 1, 10):
        x_s = np.load("../data/float_boundary_data/data_" + str(i) + "&" + str(j) +
                      "/data_" + str(i) + "&" + str(j) + "_1000_part1.npz")["x_train"]
        # x_train = np.append(x_train, x_s[0][:6000].reshape(6000, img_rows, img_cols), axis=0)
        # x_test = np.append(x_test, x_s[0][6000:].reshape(1000, img_rows, img_cols), axis=0)
        x_train = np.append(x_train, x_s[0][:222].reshape(222, img_rows, img_cols), axis=0)
        x_test = np.append(x_test, x_s[0][-44:].reshape(44, img_rows, img_cols), axis=0)
print("completed")
print(len(x_train))
print("label...")
y_train = [0 for i in range(10000)] + [1 for j in range(9990)]
y_test = [0 for i in range(2000)] + [1 for j in range(1980)]

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
# Sequential
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape,
#                  trainable=False))
# model.add(Conv2D(64, (3, 3), activation='relu', trainable=False))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu', trainable=False))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, trainable=False))
# model.add(Activation('softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# functional API
# input = Input(shape=(img_rows, img_cols, 1))
# x = Conv2D(32, kernel_size=(3, 3),
#                activation='relu',
#                input_shape=(img_rows, img_cols, 1),
#                trainable=False)(input)
# x = Conv2D(64, (3, 3), activation='relu', trainable=False)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu', trainable=False)(x)
# x = Dropout(0.5)(x)
# x = Dense(num_classes, trainable=False)(x)
# x1 = Activation('softmax')(x)
# x2 = Dense(num_classes, trainable=False)(x)
# model = Model(input, [x1, x2])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# model.summary()


# lenet-5
input = Input(shape=(img_rows, img_cols, 1))
x = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=input_shape, activation='tanh', trainable=False)(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh', trainable=False)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(120, activation='tanh', trainable=False)(x)
x = Dense(84, activation='tanh', trainable=False)(x)
y = Dense(120, activation='tanh')(x)
y = Dense(84, activation='tanh')(y)
y = Dense(2, activation='softmax')(y)
x = Dense(num_classes, activation='softmax', trainable=False)(x)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model = Model(input, y)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save("../model/freeze_test.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])


