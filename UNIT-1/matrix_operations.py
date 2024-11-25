import numpy as np

mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

print("Addition")
print(mat1 + mat2)

print("Multiplication")
print(np.dot(mat1,mat2))

print("Transpose")
print(f"for matrix 1 {mat1.T}")
print(f"for matrix 2 {mat2.T}")



"""
Output: 

Addition
[[ 6  8]
 [10 12]]
Multiplication
[[19 22]
 [43 50]]
Transpose
for matrix 1 [[1 3]
 [2 4]]
for matrix 2 [[5 7]
 [6 8]]

"""