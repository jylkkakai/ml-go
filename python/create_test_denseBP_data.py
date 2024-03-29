import random

import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(False)
filename = "../test_data/test_denseBP"
in_shape = 16
din = np.random.random(size=in_shape).astype(np.float32).reshape(1, in_shape)

input = tf.keras.layers.Input(shape=(in_shape,))
layer1 = tf.keras.layers.Dense(14, activation="sigmoid")(input)
layer2 = tf.keras.layers.Dense(12, activation="sigmoid")(layer1)
layer3 = tf.keras.layers.Dense(10, activation="sigmoid")(layer2)
layer4 = tf.keras.layers.Dense(8, activation="sigmoid")(layer3)
output = tf.keras.layers.Dense(6, activation="sigmoid")(layer4)

model = tf.keras.Model(input, output)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss="mse",
    metrics=["accuracy"],
)
wgts = model.get_weights()
target = np.random.random(size=6).astype(np.float32).reshape(1, 6)
print(target)
loss = np.zeros(1).astype(np.float32)
print(loss)
loss[0] = model.evaluate(din, target)[0]
print(loss)
print(model.evaluate(din, target))

out = model.predict(din)

model.fit(din, target)
fit_wgts = model.get_weights()
for i in range(0, 5):
    np.save(filename + "_w" + str(i) + ".npy", wgts[i * 2])
    np.save(filename + "_fw" + str(i) + ".npy", fit_wgts[i * 2])
    np.save(filename + "_b" + str(i) + ".npy", wgts[i * 2 + 1])
    np.save(filename + "_fb" + str(i) + ".npy", fit_wgts[i * 2 + 1])
np.save(filename + "_din" + ".npy", din[0])
np.save(filename + "_loss" + ".npy", loss)
np.save(filename + "_target" + ".npy", target)
np.save(filename + "_dout" + ".npy", out)
#
# fit_wgts = []
# for i in range(0, 5):
#     wgts[i * 2] = np.load(filename + "_w" + str(i) + ".npy")
#     fit_wgts.append(np.load(filename + "_fw" + str(i) + ".npy"))
#     wgts[i * 2 + 1] = np.load(filename + "_b" + str(i) + ".npy")
# din[0] = np.load(filename + "_din" + ".npy")
# loss = np.load(filename + "_loss" + ".npy")
# target = np.load(filename + "_target" + ".npy")
#
# model.set_weights(wgts)
# out = model.predict(din)
# # print(out)
# # print(target)
# # result = 0
# np.save(filename + "_dout" + ".npy", out)
# for i, v in enumerate(target[0]):
#     result += (v - out[0, i]) ** 2
#     print(result / 6)
# for i, v in enumerate(target[0]):
#     result += -(v - out[0, i])
#     print(result / target[0].shape)
# print(model.evaluate(din, target))
# print(out[0] - target[0])
# print(len(fit_wgts))
# print(fit_wgts[4])
# print()
# print(wgts[8])
# print()
# print((fit_wgts[4] - wgts[8]) * 3)
# # tloss = tf.keras.losses.mean_squared_error(target, out)
# # print(tloss)
# layer = tf.keras.layers.Dense(2, activation="relu")
# x = tf.constant([[1.0, 2.0, 3.0]])
#
# with tf.GradientTape() as tape:
#     # Forward pass
#     # model.fit(din, target)
#     y = layer(x)
#     loss = tf.reduce_mean(y**2)
#
# # Calculate gradients with respect to every trainable variable
# grad = tape.gradient(loss, layer.trainable_variables)
# print(grad)
# grad = tape.gradient(loss, layer.trainable_variables)
