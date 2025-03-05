# enter the no of inputs 
num_ip = int(input("Enter the number of inputs : "))

#Set the weights with value
w1 = 1 
w2 = 1

print("For the ", num_ip , " inputs calculate the net input using yin = x1w1 + x2w2 ")

x1 = [] 
x2 = [] 
for j in range(0, num_ip): 
    ele1 = int(input("x1 = "))
    ele2 = int(input("x2 = "))
    x1.append(ele1)
    x2.append(ele2) 
print("x1 = ",x1) 
print("x2 = ",x2)
n = x1 * w1 
m = x2 * w2

Yin = [] 
for i in range(0, num_ip):
    Yin.append(n[i] + m[i])
print("Yin = ",Yin)

#Assume one weight as excitatory and the other as inhibitory, i.e.,

Yin = []
for i in range(0, num_ip):
    Yin.append(n[i] - m[i])
print("After assuming one weight as excitatory and the other as inhibitory Yin = ",Yin)

#From the calculated net inputs, now it is possible to fire the neuron for input (1, 0) #only
#by fixing a threshold of 1, i.e., θ ≥ 1 for Y unit.
#Thus, w1 = 1, w2 = -1; θ ≥ 1

Y=[]
for i in range(0, num_ip):
    if(Yin[i]>=1):
        ele= 1
        Y.append(ele)
    if(Yin[i]<1):
        ele= 0
        Y.append(ele) 
print("Y= ",Y)



# Enter the number of inputs : 4
# For the  4  inputs calculate the net input using yin = x1w1 + x2w2 
# x1 = 0
# x2 = 0
# x1 = 0
# x2 = 1
# x1 = 1
# x2 = 0
# x1 = 1
# x2 = 1
# x1 =  [0, 0, 1, 1]
# x2 =  [0, 1, 0, 1]
# Yin =  [0, 1, 1, 2]
# After assuming one weight as excitatory and the other as inhibitory Yin =  [0, -1, 1, 0]
# Y=  [0, 0, 1, 0]