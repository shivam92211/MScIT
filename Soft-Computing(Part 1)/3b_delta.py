import numpy as np
import time

np.set_printoptions(precision=2)

x = np.zeros((3,))
weights = np.zeros((3,))
desired = np.zeros((3,))
actual = np.zeros((3,))

for i in range(0, 3):
    x[i] = float(input("Initial inputs:"))

for i in range(0, 3):
    weights[i] = float(input("Initial weights:"))

for i in range(0, 3):
    desired[i] = float(input("Desired output:"))

learning_rate = float(input("Enter learning rate:"))

actual = x * weights

print("actual", actual)
print("desired", desired)

while True:
    if np.array_equal(desired, actual):
        break  # no change
    else:
        for i in range(0, 3):
            weights[i] = weights[i] + learning_rate * (desired[i] - actual[i])
        actual = x * weights

        print("weights", weights)
        print("actual", actual)
        print("desired", desired)
        print("*" * 30)

print("Final output")
print("Corrected weights", weights)
print("actual", actual)
print("desired", desired)


# Enter weights
# Weight w11=1
# weight w12=-1
# Weight w21=-1
# weight w22=1
# weight v1=1
# weight v2=1
# Enter Threshold Value
# theta=1
# z1 [ 0 -1  1  0]
# z2 [ 0  1 -1  0]
# yin [0. 1. 1. 0.]
# Output of Net
# y [0 1 1 0]
# z [0 1 1 0]
# McCulloch-Pitts Net for XOR function
# Weights of Neuron Z1
# 1
# -1
# weights of Neuron Z2
# -1
# 1
# weights of Neuron Y
# 1
# 1
# Threshold value
# 1