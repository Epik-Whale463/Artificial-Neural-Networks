import numpy as np

# Input matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Transformation: Scaling and rotation
scaling_factor = 3
rotation_matrix = np.array([[0, 1], [-1, 0]])  # -90-degree rotation

transformed_matrix1 = scaling_factor * matrix1
transformed_matrix2 = np.dot(rotation_matrix, matrix2)

print("Transformed Matrix 1:\n", transformed_matrix1)
print("Transformed Matrix 2:\n", transformed_matrix2)


"""
Transformed Matrix 1:
 [[ 3  6]
 [ 9 12]]
Transformed Matrix 2:
 [[ 7  8]
 [-5 -6]]

 
"""