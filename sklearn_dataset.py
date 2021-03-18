from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
Y = boston.target

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)
# print(X_train, X_test, Y_train, Y_test)

import numpy as np
X = np.random.uniform(0.0, 1.0, size=(10, 2))
Y = np.random.choice(('Male', 'Female'), size=(10))
print(X[0], Y[0])

X = np.array([[1, 2, 3], [4, 5, 6]])
print('x:\n', X)