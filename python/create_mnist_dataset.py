import random

import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train[:20])
np.save("../test_data/mnist_x_train.npy", x_train.astype(np.float32))
np.save("../test_data/mnist_y_train.npy", y_train.astype(np.float32))
np.save("../test_data/mnist_x_test.npy", x_test.astype(np.float32))
np.save("../test_data/mnist_y_test.npy", y_test.astype(np.float32))
