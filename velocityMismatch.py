#!/usr/bin/env python3

"""Computes mismatch of the different interpolation schemes and the reference velocity.
"""

import argparse
import functools
from multiprocessing import Pool
import numpy as np
import psutil
import sys
from threadpoolctl import threadpool_limits

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
    parser.add_argument('-N', '--Nk', help="number of grid points for integration in each direction", type=int, default=-1)
    parser.add_argument('-c', '--coreLimit', help='maximum number of cores', type=int, default=1e10)
    return parser


def evalShiftedGrid(kFrac, ksi, dList, dis_froz_min, dis_froz_max):
    Mks = ksi.k_grid(kFrac=kFrac)
    Hk = Mks.pop("H")
    dHk = ksi.dk_grid("H", kFrac=kFrac)

    keys = list(Mks.keys())
    for k in keys:
        if not k.startswith("R_"):
            continue
        vKcom = 1j *(  np.einsum("abcmk,abcknd->abcmnd", Hk, Mks[k])
                     - np.einsum("abcmkd,abckn->abcmnd", Mks[k], Hk)) + dHk
        Mks[k.replace("R", "v")] = vKcom

    keys = [k for k in Mks.keys() if k.startswith("v") or k == "p"]
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
    norm2 = np.zeros(len(keys))
    normDiff2 = np.zeros((len(keys), len(keys)))
    for i1, k1 in enumerate(keys):
        norm2[i1] = np.linalg.norm(Mks[k1][...,dList])**2
        for i2, k2 in enumerate(keys):
            normDiff2[i1, i2] = np.linalg.norm(Mks[k1][...,dList]-Mks[k2][...,dList])**2
    return keys, norm2, normDiff2


def main(seedname, dimension, enforce_hermiticity, dis_froz_min, dis_froz_max, real_wf, Nk, coreLimit):
    fname = seedname if seedname.endswith(".npz") else seedname + "_tb.npz"
    data = {k : v  for k, v in np.load(fname).items()}
    ksi = KspaceInterpolator(**data)
    if enforce_hermiticity:
        ksi.enforce_hermiticity()
    if real_wf:
        ksi.restrict_to_real_or_imag()
    dList = [ "xyz".index(d) for d in dimension]
    print(f"Comparing in {dimension}")

    Nkg = ksi.rMats['H'].shape[0:3]
    gridDim = 3
    while gridDim>=0 and Nkg[gridDim-1] == 1:
        gridDim -= 1
    if Nk >= 0:
        Nkd = max(Nkg)
        assert Nk >= Nkd
        while Nk % Nkd != 0:
            Nkd += 1
        expand_grid =  ( *((Nkd,) *gridDim), *((1,) *(3-gridDim)) )
        ksi.expand_grid(expand_grid)
        repeat = Nk // Nkd
    else:
        repeat = 2
    repeat_grid =  ( *((repeat,) *gridDim), *((1,) *(3-gridDim)) )

    kFracOffsets = (np.mgrid[0:repeat_grid[0], 0:repeat_grid[1], 0:repeat_grid[2]] / Nk).swapaxes(0, 3).reshape((-1, 3))
    with threadpool_limits(limits=1, user_api='openmp'):
        with Pool( min(coreLimit, psutil.cpu_count(logical=False)) ) as p:
            res = p.map(functools.partial(evalShiftedGrid,
                                          ksi=ksi, dList=dList,
                                          dis_froz_min=dis_froz_min, dis_froz_max=dis_froz_max),
                                          [kFrac for kFrac in kFracOffsets])

    keys, norm2, normDiff2 = res[0]
    for k, n2, nd2 in res[1::]:
        norm2 += n2
        normDiff2 += nd2
    norm = np.sqrt(norm2)
    normDiff = np.sqrt(normDiff2)
    kl = max([len(k) for k in keys]) + 1

    mvIndex = keys.index("v")
    gsSize = np.prod( np.array(repeat_grid) * ksi.rMats['H'].shape[0:3])

    header = " " * kl + " | norm" + " " * (kl-4)
    for k in keys:
        header += f" {k:{kl}}"
    print(header)
    for n, nd, k1 in zip(norm, normDiff, keys):
        line = f"{k1:{kl+1}}| {n/(gsSize)**0.5:<{kl}f}"
        for m in nd:
            line += f" {m/n:<{kl}f}"
        print(line)


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
