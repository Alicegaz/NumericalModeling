#!/usr/bin/env python
import sys
import math
import numpy as np

def dydt(x, t, beta):
    return beta*(x - 3*t - math.sin(t))

def c(ws, w, k):
    c = 1
    for idx, w_i in enumerate(ws):
        if idx!=k:
            c*=(w-w_i)/(ws[k] - w_i)
    return c

def dxdt(y, t, dt, zs, ts, ws, ro, M, T):
    dt = ts[1] - ts[0]
    g1_2 = 0
    for k in range(len(ro)):
        # TODO determine M
        if ws[k] >= y:
            g1_1 = 0
            for j in range(1, M):
                if ws[j] >= y:
                    # TODO is it ok to start from j = 1
                    ws_t = [w for w in ws if w >= y]
                    g1_1 += c(ws_t, (ws[j - 1] + ws[j]) / 2, k) * (ws[j] - ws[j - 1]) * ro[k]

            g1_2 += g1_1

    g2_1 = 0
    g2_2 = 0
    for k in range(len(zs)):
        if t + dt <= T:
            g2_1 += c(ts, t + dt, k) * zs[k]
            g2_2 += c(ts, t, k) * zs[k]
        else:
            break
    g2_3 = g2_1 - g2_2
    x = g1_2 * g2_3
    return x

def s(t):
    return 3*t+math.sin(t)

def update_y(f, t, x, dt, beta):
    k1 = f(x, t, beta)*dt
    k2 = f(x+k1/2, t+dt/2, beta)*dt
    k3 = f(x+k2/2, t+dt/2, beta)*dt
    k4 = f(x+k3, t+dt, beta)*dt
    return k1+2*k2+2*k3+k4

def update_x(f, t, y, dt, zs, ts, ws, ro, M, T):
    k1 = f(y, t, dt, zs, ts, ws, ro, M, T)*dt
    k2 = f(y+k1/2, t+dt/2, dt, zs, ts, ws, ro, M, T)*dt
    k3 = f(y+k2/2, t+dt/2, dt, zs, ts, ws, ro, M, T)*dt
    k4 = f(y+k3, t+dt, dt, zs, ts, ws, ro, M, T)*dt
    return k1+2*k2+2*k3+k4

def rungeKutta(x0, y0, T, M, beta, n, k, ws, ro, ts, zs):
    #TODO: do we need ts or to and dt
    #number of iterations using step size
    #n = (int)((x-x0)/h)
    Y =  []
    X = []
    X[0] = x0
    Y[0] = y0
    #TODO: check if all t and dt are the same for different functions
    #TODO: precalculate dt
    #TODO: determine the length of ts
    t0 = 0
    dt = int(T/n)
    t = 0
    times = []
    times.append(t0)
    for i in range(1, n+1):
        #TODO: implement runge kutta scheme
        #y = dydt(X[i-1], t, beta)
        y_i = Y[i-1]+1/6*update_y(dydt, t, X[i-1], dt, beta)
        Y.append(y_i)
        #x = dxdt(M, y, T, t, dt, zs, ts, ws, ro)
        #t, y, dt, zs, ts, ws, ro, M, T
        x_i = X[i-1]+1/6*update_x(dxdt, t, Y[i-1], dt, zs, ts, ws, ro, M, T)
        X.append(x_i)
        t+=dt
        times.append(t)
    times = times[:-1]
    return X, Y, times

def conv(X, T, eps):
    eps_pred = abs((X[T]-s(T))/s(T))
    if eps_pred<=eps*10:
        return True
    else:
        return False

def main(argv):
    f = open(argv[0], "r")
    f2 = open(argv[1], "w")
    print(argv[0], argv[1])
    f1 = f.readlines()
    print(f1)
    z = []
    ro = []
    str1 = list(map(float, f1[0].split(' ')))
    eps, T = int(str1[0]), str1[1]
    str2 = list(map(float, f1[1].split(' ')))
    n, k = int(str1[0]), int(str1[1])
    for x in f1[2:]:
        z.append(np.array(list(map(float, x.split(' ')))))
    z = np.stack(z)
    zs = z[::2]
    ts = z[1::2]

    for x in f1[3:]:
        ro.append(np.array(list(map(float, x.split(' ')))))
    ro = np.stack(ro)
    ws = ro[1::2]
    ro = ro[::2]

    M = n + n / 2

    betas = [0.01, 1e-3, 1e-5, 1e-8, 10, 5, 1, 0.1]
    x0 = 0
    y0 = 0
    for beta in betas:
        X, Y, Ts = rungeKutta(x0, y0, T, M, beta, n, k, ws, ro, ts, zs)
        converge = conv(X[-1], T, eps)
        if converge:
            break

    for t, x, y in zip(Ts, X, Y):
        f2.write(str(t) + " " + str(x) + " " + str(y) + " ")
        # f2.write("\n")
