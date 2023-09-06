import numpy as np
import math as M

# 두 점의 좌표
global x, y
x = int(input())
y = int(input())

def distance(i, j):
    dtn = M.sqrt(M.pow(x-i, 2)+M.pow(y-j, 2))
    return dtn

# main
arr = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        d = distance(i, j)
        if d > 3 and d < 5:
            arr[i][j] = 1
print(arr)
