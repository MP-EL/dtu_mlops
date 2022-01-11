import os
import numpy as np

N_train = 25000

base_dir = os.getcwd()
dir = "/data/processed/train.npz"
tmp = np.load(base_dir + dir)
images = tmp['images']
labels = tmp['labels']

assert len(images) == N_train
assert np.shape(images) == (N_train, 28, 28), "Dataset has incorrect shape"
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in numbers:
    assert i in labels, "{0} is not represented in the labels".format(i)

