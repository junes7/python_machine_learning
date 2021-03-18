import numpy as np
X = np.random.uniform(0.0, 1.0, size=(10, 2))
Y = np.random.choice(('Male', 'Female'), size=(10))
print(X[0], Y[0])

X = np.array([[1, 2, 3], [4, 5, 6]])
print('x:\n', X)
