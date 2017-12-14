import numpy as np
import sys

if __name__ == "__main__":
    todo = sys.argv[1]
    A = np.load('samples.npy')
    if todo == "read":
        print(A)
    elif todo == "reset":
        B = np.zeros((2,11))
        B[0] = A[1]
        B[1] = A[1]
        B[0][-1] = 1
        B[1][-1] = 1
        np.save("samples.npy", B)
        print(B)
