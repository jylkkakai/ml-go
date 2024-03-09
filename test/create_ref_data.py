import tensorflow as tf
import numpy as np

h = 20
w = 30
c = 32

in_data = np.random.randint(-128, 127, size=(h, w, c))

with open('in.txt', 'w') as outfile:
    for slice in in_data:
        np.savetxt(outfile, slice, fmt='%i')
        print(slice)

