#!/usr/bin/env python
import sys
from scripts.multigrid_solver import *

def main(argv):
    f2 = open(argv[1], "w")
    f, M, eps, G = read_input_file(argv[0])
    x, coords = poisson2d_run(G, M, f, eps)
    print(x)
    it = 0
    f2.write(str(M) + " ")
    for c, u in zip(coords, x):
        f2.write(str(c[0]) + " " + str(c[1]) + " " + str(u) + " ")
        if ((it % 4 * M) == 0) and (it > 0):
            f2.write("\n")
        it += 1
    #f2 = open(argv[1], "w")

if __name__=="__main__":
    main(sys.argv[1:])
