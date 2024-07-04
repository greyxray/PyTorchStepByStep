import numpy as np

# Generate datapoints
np.random.seed(42)

N = 100

true_w, true_b = 2, 1
x = np.random.rand(N, 1)  # matrix [N, 1]
epsilon = 0.1 * np.random.randn(N, 1)  # noise from normal distribution

y = true_b + true_w * x + epsilon # y true

print(f"Done {__file__.__repr__()}")
