import numpy as np

# Generate datapoints
np.random.seed(43)

N = 1000

true_w, true_b = 2, 1

epsion = 0.1 * np.random.rand(N,1)  # noise
x = np.random.rand(N, 1) # matrix [N, 1]
y = true_w * x + true_b + epsion # y true

# Split
split_train_ratio = 0.8

indx = np.arange(N)
np.random.shuffle(indx)

x_train = x[indx[: round(N * split_train_ratio)]]
y_train = y[indx[: round(N * split_train_ratio)]]

x_val = x[indx[round(N * split_train_ratio) :]]
y_val = y[indx[round(N * split_train_ratio) :]]

print(f"Done {__file__.__repr__()}")
