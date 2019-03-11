import numpy as np
from NumericalModeling.scripts.matrix_generator import *
import sys

def read_input_file(input_path):
    f = open(input_path, "r")
    f1 = f.readlines()
    str1 = list(map(float, f1[0].split(' ')))

    f = float(str1[1])
    M = int(str1[1])
    eps = float(str1[2])

    G = []

    for x in f1[1:]:
        G.append(np.array(list(map(float, x.split(' ')))))
    G = np.stack(G)

    return f, M, eps, G

def smoother(A, b, x0=None, eps=1e-10, max_iter=1000):
    if A.shape[0]!=A.shape[1]:
        x = -1
        iter_n = 0
        return x, iter_n
    n = A.shape[0]
    iter_n = 0
    err=1
    if x0 is not None:
        x = x0
    else:
        x = np.zeros(n)+eps


    while ((err>=eps) and (iter_n<max_iter)):
        x0 = x
        for i in range(n):
            x[i] = b[i]-A[i, :]@x+A[i, i]*x[i]
            x[i] = x[i]/A[i, i]
        iter_n+=1
        err = max(abs(x0-x))
    return x, iter_n

def vcycle(level, A_list, R_list, b, x0, direct, PR_coef, pre_steps=1, pos_steps=1):
    A = A_list[level]
    n = b.shape[0]

    #solve directly
    if (n<=direct):
        x = np.linalg.solve(A, b)
        return x
    x, _ = smoother(A, b, x0=x0, eps=1e-14, max_iter=pre_steps)

    R = R_list[level]
    P = R.T * PR_coef
    coarse_n = R.shape[0]
    print("level {}, A {}, b {}, R {}, x {}".format(level, A.shape, len(b), R.shape, x.shape))
    #compute residual
    r = b - A@x
    #restrict (project) to coarse grid
    r_H = R@r

    x0 = np.zeros(coarse_n)
    e_H = vcycle(level+1, A_list, R_list, r_H, x0, direct, PR_coef, pre_steps, pos_steps)

    #interpolate error to fine grid and correct
    x = x + P@e_H

    #apply post smoothing
    x, _ = smoother(A, b, eps=1e-14, max_iter=pos_steps, x0=x)
    return x


def multigrid_solver(A, b, pre_steps=1, pos_steps=1, rn_tol=1e-10):
    pre_step = 1
    pos_step = 1

    n = A.shape[0]
    x = np.zeros(n)
    rn = np.linalg.norm(b, 2)

    vcycle_cnt = 0
    res_norm = []
    res_norm.append(rn)
    rn_stop = rn*rn_tol

    PR_coef = 4
    direct = 49

    A_list, R_list, max_level = build_multigrid(A, direct)

    while (rn > rn_stop):
        x = vcycle(1, A_list, R_list, b, x, direct, PR_coef, pre_steps=pre_step, pos_steps=pos_step)
        r = b - A@x
        rn = np.linalg.norm(r, 2)
        res_norm.append(rn)
    return x, vcycle_cnt, res_norm

def poisson2d_run(G, M, f, eps):
    A, b, coords = build_matrix_poisson(G, M, f)
    x, vc_cnt, res_norm = multigrid_solver(A, b, rn_tol=eps)
    return x, coords

#def main(argv):
#    M, eps, f, G = read_input_file(argv[0])
#    x = poisson2d_run(G, M, f, eps)
#    #f2 = open(argv[1], "w")

#if __name__=="__main__":
#    main(sys.argv[1:])


