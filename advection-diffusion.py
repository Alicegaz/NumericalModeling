#!/usr/bin/env python
from __future__ import division

import math
# os.environ["MKL_NUM_THREADS"] = "20"
# os.environ["NUMEXPR_NUM_THREADS"] = "20"
# os.environ["OMP_NUM_THREADS"] = "20"

import os
#import mkl

#mkl.set_num_threads(30)

import numpy as np
import sys

recs = [(68, 11, 98, 29),
(68, 29, 86, 89),
(68, 110 , 86, 128),
(68, 164 , 98, 182),
(68, 182 , 86, 242),
(68, 263 , 86, 281),
(137 , 11, 155 , 29),
(137 , 50, 155, 128),
(125, 110, 137, 128),
(137, 164, 155, 182),
(137, 203, 155, 281),
(125, 263, 137, 281),
(191 , 11, 221 , 29),
(191 , 29, 209 , 89),
(191, 110, 209, 128),
(191, 164, 221, 182),
(191, 182, 209, 242),
(191, 263, 209, 281),
(260 , 11, 278 , 29),
(260 , 50, 278, 128),
(248, 110, 272, 128),
(260, 164, 278, 182),
(260, 203, 278, 281),
(248, 263, 260, 281)]
recs = list(map(lambda x: list(map(lambda y: y - 1, x)), recs))

recs = np.array(recs)


def FindPoint(x, y):
    for rec in recs:
        x1, y1, x2, y2 = rec
        if (x > x1 and x < x2 and
                    y > y1 and y < y2):
            return True
    return False


def is_border(x, y):
    for rec in recs:
        x1, y1, x2, y2 = rec
        if (((x > x1) and (x < x2)) and ((y1 == y) or (y2 == y))) \
                or (((y > y1) and (y < y2)) and ((x1 == x) or (x2 == x))):
            return True
        else:
            return False


def coef_in_range(lamda, k, h):
    return (abs(lamda) / k) <= (2 / h)


def converged(X, eps):
    X = np.array(X)
    if X.shape[0] < 2:
        return False
    diff1 = (X[-1] - X[-2])
    conv = np.sum(np.abs(diff1)) <= eps
    return conv


def iteration(A, B, eps):
    reciprocal = lambda x: 1/x if x>0 else 0
    Z = np.diag(list(map(reciprocal, np.diag(A))))
    #Z = np.diag((np.reciprocal(np.diag(A))))
    eigen = np.linalg.eigvals(A)
    lambda_min = eigen.min().real
    lambda_max = eigen.max().real
    tao = 2 / (lambda_min + lambda_max)
    X = [np.matmul(Z, B)]

    while not converged(X, eps):
        print("x", X[-1])
        print("t", tao * (np.matmul(A, X[-1]) - B))
        X_new = X[-1] - tao * (np.matmul(A, X[-1]) - B)
        X.append(X_new)
    return X[-1], len(X)


## TODO: change coordinates recs
## TODO: reverce x and y
def build_matrix(M=300, lamda1=1, lamda2=0, k=0.5, step_size=1):
    u = np.zeros((M ** 2, M ** 2))
    b = np.zeros(M ** 2)
    coords = []
    for i, y in enumerate(np.arange(0, M, step_size)):
        for j, x in enumerate(np.arange(0, M, step_size)):
            coords.append((x, y))
            A = np.zeros((M, M), dtype=np.float32)
            if not FindPoint(y, x):
                A[i, j] = 4
                h = step_size
                r1 = (lamda1 * h) / (2 * k)
                r2 = (lamda2 * h) / (2 * k)
                # we do not approximate by the border points
                if not FindPoint(x, y - step_size) and \
                        not is_border(x, y - step_size) and y > 0:
                    A[i, j - 1] = -1 * (1 + r2)
                if not FindPoint(x, y - step_size) and \
                        not is_border(x, y - step_size) and y < M - 1:
                    A[i, j + 1] = -1 * (1 - r2)
                if not FindPoint(x - step_size, y) and \
                        not is_border(x - step_size, y) and x > 0:
                    A[i - 1, j] = -1 * (1 + r1)
                if not FindPoint(x + step_size, y) and \
                        not is_border(x + step_size, y) and x < M - 1:
                    A[i + 1, j] = -1 * (1 - r1)

                u_line = A.ravel()  # M^2
                u[M * i + j, :] = u_line
                if x == 0:  # g3
                  b[M * i + j] += 1
    return u, b, coords

def main(argv):
    #test_iteration_method()
    f = open(argv[0], "r")
    f2 = open(argv[1], "w")
    f1 = f.readlines()
    str1 = list(map(float, f1[0].split(' ')))
    eps = int(str1[0])
    step_size = 5
    M = int(300 / step_size)
    lamda1 = 1
    lamda2 = 0
    #k = 0.5
    #k>=lamda1/(2/step_size)
    k = math.ceil(lamda1/(2/step_size))
    h = 5
    assert (coef_in_range(lamda1, k, h) \
            and coef_in_range(lamda2, k, h))

    A, B, coords = build_matrix(step_size=step_size, M=M)
    print(type(A[0, 0]))
    np.savetxt('A.out', A, delimiter=',')
    np.savetxt('B.out', B, delimiter=',')
    print(A.shape)
    print(B.shape)
    print(A)
    print("\n")
    print(B)
    x = iteration(A, B, eps)
    f2.write(str(M)+" ")
    it = 0
    for c, u in zip(coords, x):
        f2.write(str(c[0]) + " " + str(c[1]) + " " + str(u)+" ")
        if ((it % 4*M) == 0) and (it > 0):
            f2.write("\n")
        it += 1


if __name__=="__main__":

    main(sys.argv[1:])