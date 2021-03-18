import numpy as np
from scipy import sparse
eye = np.eye(4)
print("NumPy 배열:\n", eye)

sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy의 CSR행렬:\n", sparse_matrix)