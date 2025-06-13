import tensorflow as tf

# Creating tensors with different shapes and data types
t1 = tf.constant([1, 2, 3], dtype=tf.int32)  # 1D tensor (vector)
t2 = tf.constant([[1.5, 2.5], [3.5, 4.5]], dtype=tf.float32)  # 2D tensor (matrix)

# Performing basic tensor operations
add_result = tf.add(t1, 2)  # Adding a scalar
sub_result = tf.subtract(t1, 1)  # Subtracting a scalar
mul_result = tf.multiply(t1, 2)  # Element-wise multiplication
div_result = tf.divide(t1, 2)  # Element-wise division

# Reshaping, slicing, and indexing tensors
t3 = tf.reshape(t2, [4, 1])  # Reshape to a column vector
slice_result = t2[:, 1]  # Extracting second column
index_result = t1[0]  # Extracting first element

# Performing matrix multiplication
mat1 = tf.constant([[1, 2],
                     [3, 4]], dtype=tf.float32)
mat2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
mat_mul_result = tf.matmul(mat1, mat2)  # Matrix multiplication

# Finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = tf.linalg.eig(mat1)

# Printing results
print("Addition:", add_result.numpy())
print("Subtraction:", sub_result.numpy())
print("Multiplication:", mul_result.numpy())
print("Division:", div_result.numpy())
print("Reshaped tensor:", t3.numpy())
print("Sliced tensor:", slice_result.numpy())
print("Indexed element:", index_result.numpy())
print("Matrix multiplication result:\n", mat_mul_result.numpy())
print("Eigenvalues:\n", eigenvalues.numpy())
print("Eigenvectors:\n", eigenvectors.numpy())

