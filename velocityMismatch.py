#!/usr/bin/env python3

"""Computes mismatch of the different interpolation schemes and the reference velocity.
"""

import argparse
import numpy as np
import sys


from KspaceInterpolator import KspaceInterpolator


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Computes the normalized mismatch between the different kinds of velocity and momentum operators""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname')
    parser.add_argument('-d', '--dimension', help='dimension', type=int)
    parser.add_argument('-H', '--enforce_hermiticity', help='make all all matrix elements in BZ Hermitian', action='store_true')
    parser.add_argument('-R', '--real_wf', help='neglect imaginary part of Wannier functions', action='store_true')
    return parser

def main(seedname, dimension, enforce_hermiticity, real_wf):
    fname = seedname if seedname.endswith(".npz") else seedname + "_tb.npz"
    data = {k : v  for k, v in np.load(fname).items()}
    ksi = KspaceInterpolator(**data)    
    if enforce_hermiticity:
        ksi.enforce_hermiticity()
    if real_wf:
        ksi.restrict_to_real_or_imag()
    Nk = ksi.rMats['H'].shape[0:3]
    if dimension is None:
        dimension = 3
        while dimension>=0 and Nk[dimension-1] == 1:
            dimension -= 1
    print(f"Comparing in {dimension}D")
    # expand to double-sized grid, which completely contains [H, A] in k-space (convolution)
    repeat_grid =  ( *((2,) *dimension), *((1,) *(3-dimension)) )
    Mks = ksi.k_grid(repeat=repeat_grid)
    Hk = Mks.pop("H")
    dHk = ksi.dk_grid("H", repeat=repeat_grid)
    gsSize = np.prod(Hk.shape[0:3])

    for k in data.keys():
        if not k.startswith("R_"):
            continue
        vKcom = 1j *(  np.einsum("abcmk,abcknd->abcmnd", Hk, Mks[k])
                     - np.einsum("abcmkd,abckn->abcmnd", Mks[k], Hk)) + dHk
        Mks[k.replace("R", "v")] = vKcom

    keys = [k for k in Mks.keys() if k.startswith("v") or k == "p"]
    kl = max([len(k) for k in keys]) + 1
    header = " " * kl + " | norm" + " " * (kl-3)
    for k in keys:
        header += f" {k:{kl+1}}"
    print(header)
    for k1 in keys:
        norm = np.linalg.norm(Mks[k1][...,0:dimension])
        line = f"{k1:{kl+1}}| {norm/(gsSize)**0.5:<{kl}f}"
        for k2 in keys:
            mismatch = np.linalg.norm(Mks[k1][...,0:dimension]-Mks[k2][...,0:dimension]) / norm
            line += f"  {mismatch:<{kl}f}"
        print(line)
 


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
