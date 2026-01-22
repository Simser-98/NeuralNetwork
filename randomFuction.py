import math
import random

def function(x, y):
    return (y + 5 * math.sin(x) + math.cos(y) / 34.5) * math.tan(x / 34.5)

with open("file.txt", "w") as f:

    for i in range(1000):
        X = random.randint(1, 10000)
        Y = random.randint(1, 10000)
        Z = function(X, Y)
        f.write("Input: "+ str(X) + " " + str(Y) + "\n")
        f.write("Output: "+ str(function(X, Y)) + "\n")