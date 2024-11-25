import time
import numpy as np

size  = 500
mat1 = np.random.rand(size,size)
mat2 = np.random.rand(size,size)


# Nested loop multilpication
start = time.time()
result_nested = [[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)] for row in mat1]
end = time.time()
print("Nested Loop Time:", end - start)


# NumPy multiplication
start = time.time()
result_numpy = np.dot(mat1, mat2)
end = time.time()
print("NumPy Time:", end - start)


"""
Output:

Nested Loop Time: 11.496230363845825
NumPy Time: 0.0059528350830078125


"""