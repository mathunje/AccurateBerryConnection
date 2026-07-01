#!/usr/bin/env python3


""" Computes the singular values of overlap matrices and stores it."""

import argparse
import numpy as np
from multiprocessing import Pool
import psutil
import scipy
from threadpoolctl import threadpool_limits

def sigmaSVD(M):
    return scipy.linalg.svd(M)[1]

def main(seedname, outfname):
    if seedname.endswith(".npz"):
        seedname = seedname[:-4]
    if outfname is None:
        outfname = seedname + "_singularValues"
    data = np.load(seedname + ".npz")
    M = data['M']
    with threadpool_limits(limits=1, user_api='openmp'):
        with Pool(psutil.cpu_count(logical=False)) as p:
            res = p.map(sigmaSVD, [M.reshape(-1, *M.shape[4:])[i] for i in range(np.prod(M.shape[0:4]))])
    singularValues = np.array(res).reshape(M.shape[0:-1])
    np.savez_compressed(outfname, sigma=singularValues, lattice=data['lattice'], bvec=data['bvec'])


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Computes singular values of overlap matrices (from parsed .npz file)""",
                        epilog="from MT")
    parser.add_argument('seedname', help='seedname (reads seedname.npz)')
    parser.add_argument('-o', '--output', help='output file name', dest='outfname')
    return parser

if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
