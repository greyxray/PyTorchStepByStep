import numpy as np

# Generate datapoints
np.random.seed(43)

N = 1000

true_w, true_b = 2, 1

epsion = 0.1 * np.random.rand(N,1)  # noise
x = np.random.rand(N, 1) # matrix [N, 1]
y = true_w * x + true_b + epsion # y true

print(f"Done {__file__.__repr__()}")
