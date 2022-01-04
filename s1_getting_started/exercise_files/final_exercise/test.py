from data import mnist
from operator import itemgetter

test, train = mnist()

for (images, labels) in test:
    print(images)