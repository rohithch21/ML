import numpy as np

N = int(input())
A = [[pow(i,2)] for i in range(N*N)]
A = np.array(A)
print(A)
A = np.reshape(A,(N,N))
print(A)
for ele in range(N):
    c = np.array((input().split()))
    A[ele] = c
print(A)