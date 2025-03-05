import math

a0 = -1
t = -1

w10 = float(input("Enter weight for the first network: "))
b10 = float(input("Enter bias for the first network: "))
w20 = float(input("Enter weight for the second network: "))
b20 = float(input("Enter bias for the second network: "))

c = float(input("Enter learning coefficient: "))

n1 = float(w10 * a0 + b10)
a1 = math.tanh(n1)
n2 = float(w20 * a1 + b20)
a2 = math.tanh(n2)

e = t - a2
s2 = -2 * (1 - a2 * a2) * e
s1 = (1 - a1 * a1) * w20 * s2

w21 = w20 - (c * s2 * a1)
w11 = w10 - (c * s1 * a0)
b21 = b20 - (c * s2)
b11 = b10 - (c * s1)

print("The updated weight of the first network w11 =", w11)
print("The updated weight of the second network w21 =", w21)
print("The updated bias of the first network b11 =", b11)
print("The updated bias of the second network b21 =", b21)


# Enter weight for the first network: 12
# Enter bias for the first network: 35
# Enter weight for the second network: 23
# Enter bias for the second network: 45
# Enter learning coefficient: 11
# The updated weight of the first network w11 = 12.0
# The updated weight of the second network w21 = 23.0
# The updated bias of the first network b11 = 35.0
# The updated bias of the second network b21 = 45.0