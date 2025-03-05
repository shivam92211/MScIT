import math

def sigmoid(x):
  return 1/1+math.exp(-x)

n = int(input("enter the no of element:"))

print("your inputs", n)

inp = []

for i in range(0,n):
  ele = float(input("enter x values: "))
  inp.append(ele)

print(inp)

weight = []

for i in range(0,n):
  ele = float(input("enter weight values: "))
  weight.append(ele)

print(weight)

y = []

for i in range(0,n):
  yin = inp[i]*weight[i]
  y.append(yin)

z = round(sum(y))

print(round(sum(y),3))



print(sigmoid(z))


# enter the no of element:3
# your inputs 3
# enter x values: 0.3
# enter x values: 0.5
# enter x values: 0.6
# [0.3, 0.5, 0.6]
# enter weight values: 0.2
# enter weight values: 0.1
# enter weight values: -0.3
# [0.2, 0.1, -0.3]
# -0.07
# 2.0