#!/usr/bin/env python3

"""Computes mismatch of the different interpolation schemes and the reference velocity.
"""

import argparse
import numpy as np
import sys


import atu
from KspaceInterpolator import KspaceInterpolator


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Computes the normalized mismatch between the different kinds of velocity and momentum operators""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname')
    parser.add_argument('-d', '--dimension', help='dimension', default='xyz')
    parser.add_argument('-H', '--enforce_hermiticity', help='make all all matrix elements in BZ Hermitian', action='store_true')
    parser.add_argument('-R', '--real_wf', help='neglect imaginary part of Wannier functions', action='store_true')
    parser.add_argument('-dmin', '--dis_froz_min', help='minimum of disentanglement window [eV]', type=float, default=-np.inf)
    parser.add_argument('-dmax', '--dis_froz_max', help='maximum of disentanglement window [eV]', type=float, default=np.inf)
    return parser

def main(seedname, dimension, enforce_hermiticity, dis_froz_min, dis_froz_max, real_wf):
    fname = seedname if seedname.endswith(".npz") else seedname + "_tb.npz"
    data = {k : v  for k, v in np.load(fname).items()}
    ksi = KspaceInterpolator(**data)    
    if enforce_hermiticity:
        ksi.enforce_hermiticity()
    if real_wf:
        ksi.restrict_to_real_or_imag()
    Nk = ksi.rMats['H'].shape[0:3]
    dList = [ "xyz".index(d) for d in dimension]
    print(f"Comparing in {dimension}")
    # expand to double-sized grid, which completely contains [H, A] in k-space (convolution)
    repeat_grid = 2 #( *((2,) *dimension), *((1,) *(3-dimension)) )
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

    # overwrite with velocity matrix elements in Hamiltonian basis, where all matrix elements
    # that belong to bands outside the energy window are zeroed.
    if np.isfinite(dis_froz_min) or np.isfinite(dis_froz_max):
        df_min = atu.from_eV(dis_froz_min)
        df_max = atu.from_eV(dis_froz_max)
        Ek, Uk = np.linalg.eigh(Hk)
        Ekd = np.repeat(Ek[...,None, :], Ek.shape[-1], axis=-2)
        cropped_vH = []
        for k in keys:
            vH = np.einsum("xyzba,xyzbc...,xyzcd->xyzad...", Uk.conj(), Mks[k][...,dList], Uk)
            vH[Ek<df_min] = 0
            vH[Ek>df_max] = 0
            vH[Ekd<df_min] = 0
            vH[Ekd>df_max] = 0
            Mks[k] = vH
    norm = np.zeros((len(keys)))
    mismatch = np.zeros((len(keys), len(keys)))
    for i1, k1 in enumerate(keys):
        norm[i1] = np.linalg.norm(Mks[k1][...,dList])
        for i2, k2 in enumerate(keys):
            mismatch[i1, i2] = np.linalg.norm(Mks[k1][...,dList]-Mks[k2][...,dList]) / norm[i1]

    header = " " * kl + " | norm" + " " * (kl-3)
    for k in keys:
        header += f"{k:{kl+1}}"
    print(header)
    for mm, nrm, k1 in zip(mismatch, norm, keys):
        line = f"{k1:{kl+1}}| {nrm/(gsSize)**0.5:<{kl}f}"
        for m in mm:
            line += f" {m:<{kl}f}"
        print(line)


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
