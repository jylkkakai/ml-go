import numpy as np
import tensorflow as tf

h = 20
w = 30
c = 32


in_data = np.array([(0.05, 0.1)])
print(in_data.shape)
target = np.array([(0.01, 0.99)])

input = tf.keras.layers.Input(shape=(2,))
layer1 = tf.keras.layers.Dense(2, activation="sigmoid")(input)
output = tf.keras.layers.Dense(2, activation="sigmoid")(layer1)
model = tf.keras.Model(input, output)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss="mse",
    metrics=["accuracy"],
)
wgts = model.get_weights()
print(len(wgts))
print(wgts[0].shape)
wgts[0] = np.array([[0.15, 0.25], [0.2, 0.3]])
wgts[1] = np.array([0.35, 0.35])
wgts[2] = np.array([[0.4, 0.5], [0.45, 0.55]])
wgts[3] = np.array([0.6, 0.6])
print(len(wgts))
print(wgts[0].shape)
print(wgts)
model.set_weights(wgts)
np.save("../data/wgt.npy", wgts[0])
out = model.predict(in_data)

print(in_data.shape)
print(out.shape)
print(out)
loss = model.evaluate(in_data, target)
print(loss)

model.fit(in_data, target)
wgts = model.get_weights()
print(len(wgts))
print(wgts[0].shape)
igts = model.get_weights()
print(len(wgts))
print(wgts[0].shape)
print(wgts)
print(wgts[0].transpose())
