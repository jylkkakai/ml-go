import random

import numpy as np
import tensorflow as tf

dims = 1
for i in range(0, 10):

    filename = "../test_data/test_dense{}".format(i)
    in_shape = random.randint(8, 32)
    din = np.random.random(size=in_shape).astype(np.float32).reshape(1, in_shape)
    # din = np.array([random.randint(8, 32)])

    input = tf.keras.layers.Input(shape=(in_shape,))
    output = tf.keras.layers.Dense(random.randint(8, 32), activation="sigmoid")(input)
    model = tf.keras.Model(input, output)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
        loss="mse",
        metrics=["accuracy"],
    )
    wgts = model.get_weights()
    rbias = np.random.random(size=wgts[1].shape).astype(np.float32)
    wgts[1] = rbias
    model.set_weights(wgts)
    dout = model.predict(din)

    np.save(filename + "_din.npy", din[0])
    np.save(filename + "_w.npy", wgts[0])
    np.save(filename + "_b.npy", wgts[1])
    np.save(filename + "_dout.npy", dout[0])
