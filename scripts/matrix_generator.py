import numpy as np
from math import floor, sqrt
from scipy.sparse import csr_matrix

def is_inner_square_edge(x, M):
    if x/M == 0.4 or x/M == 0.6:
        return True
    else:
        return False

def build_matrix_poisson(G, M, f):
    u = np.zeros((M**2, M**2))
    b = np.full(M**2, f)
    coords = []
    for i in range(0, M):
        for j in range(0, M):
            x = i*1/M
            y = j*1/M
            coords.append((x, y))
            A = np.zeros((M, M), dtype=np.float32)
            A[i, j] = 4
            if j > 0:
                A[i, j - 1] = -1
            if j < M-1:
                A[i, j + 1] = -1
            if i > 0:
                A[i - 1, j] = -1
            if i < M-1:
                A[i + 1, j] = -1

            u_line = A.ravel()  # M^2
            u[M*i+j, :] = u_line

            if i == 0 and not is_inner_square_edge(j, M):  # g3
                b[M*i+j] += G[0, j]
            if i == M - 1 and not is_inner_square_edge(j, M):
                b[M*1+j] += G[1, j]
            if j == 0 and not is_inner_square_edge(i, M):
                b[M * i + j] += G[2, j]
            if j == M-1 and not not is_inner_square_edge(i, M):
                b[M * i + j] += G[3, j]

    return u, b, coords

def build_multigrid(A, direct):
    N = A.shape[0]
    n = floor(sqrt(N))
    #coarse_dim = floor((n-1)/2)
    #print("coarse shape", coarse_dim)

    A_list = {}
    R_list = {}

    step = 1;
    A_list[step] = A

    while (N>direct):
        coarse_dim = floor((n-1)/2)
        R = csr_matrix((coarse_dim**2, N)).toarray()
        k = 0
        for j in range(1, n, 2):
            for i in range(1, n, 2):
                fine_grid_k = (j-1)*n+i
                R[k, fine_grid_k-n-1] = 0.0625
                R[k, fine_grid_k - n] = 0.125
                R[k, fine_grid_k - n + 1] = 0.0625

                R[k, fine_grid_k - 1] = 0.125
                R[k, fine_grid_k] = 0.25
                R[k, fine_grid_k + 1] = 0.125

                R[k, fine_grid_k + n - 1] = 0.0625
                R[k, fine_grid_k + n] = 0.125
                R[k, fine_grid_k + n + 1] = 0.0625
                k+=1

        R_list[step] = R

        P = 4*R.T
        A = np.matmul(np.matmul(R, A), P)
        print("A coarse", A.shape)
        step +=1
        A_list[step] = A
        n = coarse_dim
        N = coarse_dim*coarse_dim
    return A_list, R_list, step
