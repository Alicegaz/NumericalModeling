#!/usr/bin/env python

import numpy as np

import math
import sys, getopt
from itertools import compress

def converged(Y_hat, Y, eps):
    if Y_hat.shape[0] < 2:
        return False
    diff1 = (Y_hat - Y.ravel())
    error = np.sum(np.abs(diff1))
    conv =  error <= eps
    return conv, error

def build_matrix_poisson(M, break_mode=False):
    u = np.zeros(((4*M+1)**2, (4*M+1)**2))
    b = np.full((4*M+1, 4*M+1), -4)
    Y = np.zeros((4*M+1, 4*M+1))
    u = []
    for i in range(0, 4*M+1):
        for j in range(0, 4*M+1):
            x = -1 + i / (2*M)
            y = -1 + j / (2*M)
            A = np.zeros((4*M+1, 4*M+1), dtype=np.float32)
            if (x ** 2 + y ** 2 <= 1):
                A[i, j] = 4
                y_1 = -1 + (j-1) / (2*M)
                if j > 0 and y_1**2+x**2<=1:
                    A[i, j - 1] = -1
                y_2 = -1 + (j + 1) / (2 * M)
                if j < 4*M and y_2**2+x**2<=1:
                    A[i, j + 1] = -1
                x_1 = -1 + (i-1) / (2*M)
                if i > 0 and x_1**2+y**2<=1:
                    A[i - 1, j] = -1
                x_2 = -1 + (i+1) / (2*M)
                if i < 4*(M-1) and x_2**2+y**2<=1:
                    A[i + 1, j] = -1
                Y[i, j] = 1 - (x ** 2 + y ** 2)
            else:
                b[i, j] = 0
                #TODO fill b vector
            u_line = A.ravel()  # M^2
            #u[(2*M+1)*i+j, :] = u_line
            u.append(u_line)
            #x = -1+i/M
            #y = -1+j/M
            #x = i/(2*M)
            #y = j/(2*M)
            #Y[i, j] = 1 - (x ** 2 + y ** 2)
    #u = np.array(u)
    #b = np.array(b)

    #inds = np.all(u == 0, axis=1)
    #rm_idx = list(compress(range(0, u.shape[0]), inds))
    #u = np.delete(u, rm_idx, axis=1)
    #u = u[~np.all(u == 0, axis=1)]
    #b = b.ravel()
    #Y = Y.ravel()
    #Y = np.delete(Y, rm_idx, axis=0)
    #b = np.delete(b, rm_idx, axis=0)

    return u, b.ravel(), Y.ravel()

def print_matrix(A, B):
    print("shape", np.array(A).shape)
    print([str(a).replace('.0, ', ' ') for a in np.array(A).tolist()])
    print(B)

def build_matrix_laplacce(G, M, break_mode=False):
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

def cg(A, b, tol):
    it=0; x = 0;
    r = np.copy(b); r_prev = np.copy(b)
    rho = np.dot(r,r)
    p = np.copy(r)
    x_old = 0

    while (np.sqrt(rho) > tol*np.sqrt(np.dot(b,b))) :
        it += 1
        if it == 1:
            p[:] = r[:]
        else:
            print(r_prev)
            print(np.dot(r_prev,r_prev))
            beta = np.dot(r,r)/np.dot(r_prev,r_prev)
            p = r + beta*p
        w = np.dot(A, p)
        alpha = np.dot(r,r)/np.dot(p, w)
        x_old = x
        x = x + alpha*p
        r_prev[:] = r[:]
        r = r - alpha*w
        rho = np.dot(r,r)
        if (it > 1 and np.nonzero(np.isnan(x) == True)[0].shape[0] == np.array(x).shape[0]):
            x = x_old
            break
    return x, it

def conjugate_grad(A, b, x=None):
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print('Itr:', i)
            break
        p = beta * p - r
    return x


def main(argv):
    #test_iteration_method()
    f = open(argv[0], "r")
    f2 = open(argv[1], "w")
    f1 = f.readlines()
    str1 = list(map(float, f1[0].split(' ')))
    eps = int(str1[0])
    error = 0
    for M in range(2, 5):
        A, B, Y = build_matrix_poisson(M)
        #print("y", Y)
        x = conjugate_grad(A, B)
        conv = converged(x, Y, eps)
        print("M: {}, error: {}".format(M, conv[1]))
        if conv[0]:
            print(conv[1])
            break
    print(M)
    print(x)
    print(Y)
    #print(n_iter)
    #f2.write(str(x)[1:-1])

def test_iteration_method():
    b = np.array([list(map(float, "3.5649494 3.5185026 3.4714836 3.4239174 3.3758356 3.3272777 3.2782924 3.2289386 3.1792869 3.1294215 3.0794415".split(" "))),
         list(map(float, "3.3025851 3.241773 3.1792869 3.11505 3.0489823 2.9810015 2.9110229 2.8389611 2.7647308 2.6882491 2.6094379".split(" "))),
         list(map(float, "3.5649494 3.5344901 3.5047093 3.4756977 3.4475509 3.4203681 3.3942523 3.3693087 3.3456446 3.3233676 3.3025851".split(" "))),
         list(map(float, "3.0794415 3.0294632 2.9796212 2.9300711 2.8809906 2.8325815 2.7850705 2.7387102 2.6937791 2.6505799 2.6094379".split(" ")))])
    A, B = build_matrix_laplacce(b, len("3.5649494 3.5185026 3.4714836 3.4239174 3.3758356 3.3272777 3.2782924 3.2289386 3.1792869 3.1294215 3.0794415".split(" ")))
    u, n_iter = cg(A, B, 0.0000001)
    print(u)
    print(n_iter)
    #print(n_iter)

if __name__=="__main__":

    main(sys.argv[1:])
