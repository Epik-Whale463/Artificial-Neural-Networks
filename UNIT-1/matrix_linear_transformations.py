import numpy as np

# Input matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

scaling_factor = 3
rotation_matrix = np.array([[0,-1], [1,0]]) # 90 degree rotation

scaled_matrix1 = scaling_factor * matrix1
rotated_matrix2 = np.dot(rotation_matrix,matrix2)


print("Scaled Matrix 1:\n", scaled_matrix1)
print("Rotated Matrix 2:\n", rotated_matrix2)


"""
Output: 

Scaled Matrix 1:
 [[ 3  6]
 [ 9 12]]
Rotated Matrix 2:
 [[-7 -8]
 [ 5  6]]


"""