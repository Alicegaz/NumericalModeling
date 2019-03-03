#!/usr/bin/env python

import numpy as np

import math
import sys, getopt

def converged(X, eps):
    X = np.array(X)
    if X.shape[0] < 2:
        return False
    diff1 = (X[-1] - X[-2])
    conv = np.sum(np.abs(diff1)) <= eps
    return conv

def iteration(A, B, eps):
    Z = np.diag((np.reciprocal(np.diag(A))))
    eigen = np.linalg.eigvals(A)
    lambda_min = eigen.min().real
    lambda_max = eigen.max().real
    tao = 2/(lambda_min+lambda_max)
    X = [np.matmul(Z, B)]

    while not converged(X, eps):
        X_new = X[-1] - tao*(np.matmul(A, X[-1]) - B)
        X.append(X_new)
    return X[-1], len(X)

# def build_matrix(G):
#     A = np.zeros((G.shape[1], G.shape[1]))
#     A[:, 0] = G[0]
#     A[:, A.shape[1]-1] = G[1]
#     A[0, :] = G[2]
#     A[A.shape[0]-1, :] = G[3]
#     B = np.zeros_like(A)
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             B[i-1, j-1] = 0.25*(A[i-1, j]+A[i+1, j]+A[i][j-1]+A[i][j+1])


def build_matrix(G, M, break_mode=False):
    u = np.zeros((M**2, M**2))
    b = np.zeros(M**2)
    for i in range(0, M):
        for j in range(0, M):
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

            if i == 0:  # g3
                b[M*i+j] += G[0, j]
            if i == M - 1:
                b[M*1+j] += G[1, j]
            if j == 0:
                b[M * i + j] += G[2, j]
            if j == M-1:
                b[M * i + j] += G[3, j]

    if break_mode:
        for i in range(0, M):
            for j in range(0, M):
                y = j/M
                x = i/M
                if i == 0:
                    b[M*i+j] -= 1
                if i == M-1:
                    b[M*i+j] -= (2/math.pi)*np.arctan(y)
                if j == M-1 and i!=0:
                    b[M*i+j] -= (2/math.pi)*np.arctan(1/x)
    return u, b

def is_break(G):
    M = G.shape[1]
    u_0_0_y = G[0, 0]
    u_0_1_y = G[0, M-1]
    u_1_0_y = G[1, 0]
    u_1_1_y = G[1, M-1]
    u_0_0_x = G[2, 0]
    u_1_0_x = G[2, M-1]
    u_0_1_x = G[3, 0]
    u_1_1_x = G[3, M-1]
    if u_0_0_y != u_0_0_x:
        return True
    if u_0_1_y != u_0_1_x:
        return True
    if u_1_0_y != u_1_0_x:
        return True
    if u_1_1_y != u_1_1_x:
        return True
    return False

def main(argv):
    f = open(argv[0], "r")
    f2 = open(argv[1], "w")
    print(argv[0], argv[1])
    f1 = f.readlines()
    print(f1)
    G = []
    str1 = list(map(float, f1[0].split(' ')))
    M, eps = int(str1[0]), str1[1]

    for x in f1[1:]:
        G.append(np.array(list(map(float, x.split(' ')))))
    G = np.stack(G)
    break_mode = is_break(G)
    A, B = build_matrix(G, M, break_mode=break_mode)
    x, n_iter = iteration(A, B, eps)
    print(break_mode)
    print(x)
    print(n_iter)
    f2.write(str(x)[1:-1])

def test_iteration_method():
    a = [[200, 1, 3, 2], [3, 600, 4, 7], [6, 8, 300, 5], [2, 4, 6, 500]]
    b = [6, 4, 2, 7]
    A = np.array(a, dtype=np.float32)
    B = np.array(b, dtype=np.float32)
    x, n_iter = iteration(A, B, 0.0000001)
    print(x)
    print(n_iter)

if __name__=="__main__":

    main(sys.argv[1:])
