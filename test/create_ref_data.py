import numpy as np
import tensorflow as tf

h = 20
w = 30
c = 32

# in_data = np.random.randint(-128, 127, size=(h, w, c))
in_data = np.random.randint(-128, 127, size=(1, 16))

# with open("in.txt", "w") as outfile:
#     for slice in in_data:
#         np.savetxt(outfile, slice, fmt="%i")
#         print(slice)

# in_data = np.reshape(in_data, (1, h, w, c))
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Conv2D(
#             32, 3, input_shape=(h, w, c), use_bias=False, padding="same"
#         )
#     ]
# )
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(32, input_shape=(16,), use_bias=False)]
)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
wgts = model.get_weights()
# print(type(wgts[0]))
print(wgts[0].shape)
# wgt = wgts[0].transpose((3, 0, 1, 2))
# print(wgt.shape)
out = model.predict(in_data)
# print(type(out))
# print(out.shape)

print(in_data.shape)
# print(wgt.shape)
print(out.shape)


def write_flat(filename, arr):
    with open(filename, "w") as outfile:
        for i in arr:
            for data in i:
                outfile.write(str(data) + " ")


write_flat("inpy.txt", in_data)
write_flat("wgtpy.txt", wgts[0])
write_flat("outpy.txt", out)
# def write_array(filename, arr):
#     with open(filename, "w") as outfile:
#         for i in arr:
#             for j in i:
#                 for k in j:
#                     for data in k:
#                         outfile.write(str(data) + " ")
#                     outfile.write("\n")
#         print(filename)
#
#
# write_array("in.txt", in_data)
# write_array("wgt.txt", wgt)
# write_array("out.txt", out)
