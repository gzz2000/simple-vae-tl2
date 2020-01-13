import tensorlayer as tl
import tensorflow as tf
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape = (-1, 784))

cnt_train = X_train.shape[0]
cnt_val = X_val.shape[0]

def enum_data(batch_size, X):
    for a, b in tl.iterate.minibatches(inputs = X, targets = np.arange(len(X)), batch_size = batch_size, shuffle = True):
        yield tf.concat(a, axis = 0)

def enum_train(batch_size):
    return enum_data(batch_size, X_train)

def enum_val(batch_size):
    return enum_data(batch_size, X_val)

def gray2rgb(im):
    return np.stack([im, im, im], axis = 3)
