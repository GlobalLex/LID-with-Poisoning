import random

random.seed(3)

with open('y_train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

f = open("y_train - Poisoned.txt", "w")

iterationLen = (len(lines)//100) * 70

print(iterationLen)
#Poisoning randomly
for i in range(iterationLen):
    lines[i] = lines[random.randint(0,len(lines)-1)]
    f.write("poi\n")

#Adding the rest of the file (clean Labels)
for i in range(iterationLen, len(lines)):
    f.write(lines[i])