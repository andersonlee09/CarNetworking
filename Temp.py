from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
print(X)
X = np.array(X)

print(X.shape)
print(X[0])
