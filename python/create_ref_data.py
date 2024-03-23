import random

import numpy as np

dims = 1
for i in range(0, 12):

    shape = []
    for j in range(0, dims):
        shape.append(random.randint(2, 10))

    # print(hape
    arr = np.random.random(size=shape).astype(np.float32)
    print("Shape:", arr.shape)
    filename = "../data/test_at{}".format(i)
    np.save(filename + ".npy", arr)
    index = []
    with open(filename + ".golden", "w") as file:
        for j in range(0, 10):
            index = []
            for k in range(0, len(shape)):
                index.append(random.randint(1, shape[k]) - 1)
                # print("Index:", index)

            file.write(str(index) + ";" + str(arr[tuple(index)]) + "\n")
    # index = np.random.choice(arr, len(shape), replace=False)
    # print("Index:", index)
    # print("At:", arr[tuple(index)])
    if i % 2 == 1:
        dims += 1
