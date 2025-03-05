# Python program to illustrate 'in' operator for finding common members in lists

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9]

# Using 'in' operator
for item in list1:
    if item in list2:
        print("overlapping")
        break
else:
    print("not overlapping")

# Python program to illustrate finding common members in lists without using 'in' operator

# Define a function that takes two lists
def overlapping(list1, list2):
    for i in list1:
        for j in list2:
            if i == j:
                return True
    return False

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9]

if overlapping(list1, list2):
    print("overlapping")
else:
    print("not overlapping")

# Python program to illustrate 'not in' operator

x = 24
y = 20
my_list = [10, 20, 30, 40, 50]

if x not in my_list:
    print("x is NOT present in the given list")
else:
    print("x is present in the given list")

if y in my_list:
    print("y is present in the given list")
else:
    print("y is NOT present in the given list")

