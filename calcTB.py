#!/usr/bin/env python3

"""Calculates the energies and position operator (aka Berry connection) in supercell
from parsed files and outputs the position operator and optionally the momentum operator
as npz file ( in a.u.)  or human readable format - aligned to seedname_tb.dat ( in A, eV)
    First lines conatins list of matrices in this file
    lattice
    ndegen
    blocks
"""

import argparse
import itertools
import math
from multiprocessing import Pool
import numpy as np
import psutil
import os
import scipy
from threadpoolctl import threadpool_limits

import atu
from inputParser import parse_all


class WannierCalculator:
    def __init__(self, H, M, lattice, bvec, **vMats):
        self.H = H
        self.M = M
        self.bvec = bvec
        self.lattice = lattice
        self.vMats = vMats
        self.recipLattice = 2*np.pi * np.linalg.inv(self.lattice).T
        self.nkd = self.M.shape[0:3]
        self.nk = np.prod(self.nkd)
        self.nw = self.M.shape[-1]
        self.ndegen = self.calcNdegen()
        self.wb = self.calcWeights(self.bvec)

    def k_crys2cart(self, k):
        return k @ self.recipLattice

    def k_cart2crys(self, k):
        return self.lattice @ k / (2*np.pi)

    def calcNdegen(self):
        searchSize = 2
        ndegen = {}
        n = [int(s) for s in self.nkd]
        for a, b, c in itertools.product(range(n[0]), range(n[1]), range(n[2])):
            origR = [a, b, c] @ self.lattice
            rList = []
            optR2 = np.linalg.norm(origR)
            for oa, ob, oc in itertools.product(range(-searchSize, searchSize+1), repeat=3):
                cr=(a+oa*n[0],b+ob*n[1],c+oc*n[2])
                cR2 = np.linalg.norm(cr @ self.lattice)
                if cR2 <= 0.9999 * optR2:
                    optR2 = cR2
                    rList = [cr]
                elif cR2 <= 1.0001 * optR2:
                    rList.append(cr)
            for r in rList:
                ndegen[r] = ( (a,b,c), len(rList) )
        return ndegen 
    
    def calcWeights(self, bvec):
        b = self.k_crys2cart(bvec)
        bmat = np.einsum("ia,ib->iab", b, b).reshape((b.shape[0], b.shape[1]**2))
        return np.eye(b.shape[1]).flatten() @ scipy.linalg.pinv(bmat)

    def to_Mr(self, Mk):
        Mr = np.fft.fftn(Mk, axes=(0, 1, 2), norm="forward")
        res = {}
        for Rput, (Rorig, _) in self.ndegen.items():
            res[Rput] = Mr[*Rorig]
        return res

    def write_tb_dat(self, fname, **rMatDict):
        ndegenSorted = sorted(self.ndegen.items(), key=lambda item : item[0])
        Rn, ndeg = zip(* list(map(lambda kvp: [kvp[0], kvp[1][1]], ndegenSorted)))
        with open(fname, "w") as f:
            f.write(" ".join(rMatDict.keys()) + "\n")
            for i in range(3):
                f.write(" " + "    ".join([f"{atu.to_A(self.lattice[i, j]):+.15f}" for j in range(3)]) + "\n")
            f.write(f"{self.nw:8d}\n")
            f.write(f"{len(ndeg):8d}\n")
            while len(ndeg) > 0:
                f.write("".join([f"{s:>5d}" for s in ndeg[0:15]]) + "\n")
                ndeg = ndeg[15::]
            for name, M in rMatDict.items():
                scale = 1
                if name == "H":
                    scale =  atu.to_eV(1)
                elif name.startswith("R"):
                    scale = atu.to_A(1)
                else:
                    # assume momentum / velocity units
                    scale = atu.to_eV(1) * atu.to_A(1)
                for R in Rn:
                    f.write("\n")
                    f.write("".join([f"{s:>5d}" for s in R]) + "\n")
                    Mr = M[*R]
                    for i in range(self.nw):
                        for j in range(self.nw):
                            entries = Mr[j, i].flatten().view(float)
                            f.write(f"{j+1:>5d}{i+1:>5d}  " + " ".join([f"{scale*e:+.12e}" for e in entries]) + "\n")

    def create_ksi_dict(self, **rMatDict):
        R = np.array([np.array(rc) for rc in self.ndegen.keys()])
        Rmin = np.min(R, axis=0)
        Rmax = np.max(R, axis=0)
        Rdim = Rmax - Rmin + 1
        resDict = { "RcellShift" : Rmin, "lattice" : self.lattice }
        for key, M in rMatDict.items():
            data = np.zeros((*Rdim,*M[(0, 0, 0)].shape), dtype=complex)
            for Rput, (_, ndeg) in self.ndegen.items():
                data[*(np.array(Rput)-Rmin)] = M[Rput] / ndeg
            resDict[key] = data
        return resDict


    ##############################################
    #                                            #
    # Interpolation schemes for Berry connection #
    #                                            #
    ##############################################
    # The FFT here uses the opposite sign convension compared to the manuscript 
    # "Self-consistent evaluation of the Berry connection for Wannier functions"
    # to which the equations numbers refer to
    #############################################

    def calc_MV(self):
        Mmod = self.M.copy()
        # Eq. (25)
        np.einsum("xyzsaa->xyzsa", Mmod)[:] = 1j * np.log(np.einsum("xyzsaa->xyzsa", Mmod)).imag
        # Eq. (24)
        rk = 1j * np.einsum("b,ba,xyzbmn->xyzmna", self.wb, self.k_crys2cart(self.bvec), Mmod)
        rr = np.fft.fftn(rk, axes=(0, 1, 2), norm="forward")
        rC = {}
        for Rput, (Rorig, _) in self.ndegen.items():
            rC[Rput] = rr[*Rorig]
        return rC
    
    def calc_sym(self):
        Mmod = self.M.copy()
        # Eq. (25)
        np.einsum("xyzsaa->xyzsa", Mmod)[:] = 1j * np.log(np.einsum("xyzsaa->xyzsa", Mmod)).imag
        mmnR = np.fft.fftn(Mmod, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        for Rput, (Rorig, _) in self.ndegen.items():
            # Eq. (32) in real-space
            phase = self.bvec @ Rput
            rC[*Rput] = 1j *  np.einsum("xs,xmn,x->mns", ba, mmnR[*Rorig], np.exp(-1j * np.pi * phase))
        return rC

    def calc_Lihm(self):
        # Eq. (27)
        R0 = - np.einsum("b,ba,xyzbm->ma", self.wb, self.k_crys2cart(self.bvec), np.log(np.einsum("xyzsaa->xyzsa", self.M)).imag) / self.nk
        mmnR = np.fft.fftn(self.M, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        Rfrac = R0 @ self.recipLattice.T / ( 2 * np.pi)
        RfracDiff = Rfrac[None, :, :] + Rfrac[:, None, :]
        for Rput, (Rorig, _) in self.ndegen.items():
            # Eq. (28)
            phase = np.einsum("bx,mnx->bmn", self.bvec, np.array(Rput)[None, None, :] - RfracDiff)
            rC[*Rput] = 1j *  np.einsum("xs,xmn,xmn->mns", ba, mmnR[*Rorig], np.exp(-1j * np.pi * phase))
        for n in range(self.nw):
            rC[(0, 0, 0)][n, n] = R0[n]
        return rC

    def calc_log(self):
        # Eq. (40)
        nb = self.bvec.shape[0]
        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                res = p.map(scipy.linalg.logm, [self.M.reshape(self.nk, nb, self.nw, self.nw)[i//nb, i%nb] for i in range(nb*self.nk)])
        logMmn = np.array(res).reshape(self.M.shape)
        mmnR = np.fft.fftn(logMmn, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        for Rput, (Rorig, _) in self.ndegen.items():
            phase = self.bvec @ Rput
            rC[*Rput] = 1j *  np.einsum("xs,xmn,x->mns", ba, mmnR[*Rorig], np.exp(-1j * np.pi * phase))
        return rC

    def calc_clog(self, maxIterations=20):
        # Eqs. (41) - (45)
        nb = self.bvec.shape[0]
        nw = self.M.shape[-1]
        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                res = p.map(scipy.linalg.logm, [self.M.reshape(self.nk, nb, self.nw, self.nw)[i//nb, i%nb] for i in range(nb*self.nk)])
        logMmn = np.array(res).reshape(self.M.shape)
        logMkb = logMmn
        bDk = logMkb
        phaseFac = np.zeros((*logMkb.shape[0:3], nb), dtype=complex)
        for Rput, (Rorig, nc) in self.ndegen.items():
            phaseFac[Rorig] += np.exp(1j * np.pi * self.bvec @ Rput) / nc
        error = np.linalg.norm(0.5*(bDk-np.swapaxes(bDk, -1, -2).conj())-logMkb)
        it = 0
        while it < maxIterations:
            print(f"\titeration {it}: error={error}")
            Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
            bDk1 = np.fft.ifftn(np.einsum("xyzbmn,xyzb->xyzbmn", Rb, phaseFac), axes=(0, 1, 2), norm="forward")
            bDk2 = bDk
            bDk3 = np.fft.ifftn(np.einsum("xyzbmn,xyzb->xyzbmn", Rb, np.conj(phaseFac)), axes=(0, 1, 2), norm="forward")
            # Magnus expansion of 4th order of path ordered integral - Eq. (42)
            omegaDkb = 1/6 * (bDk1 + 4 * bDk2 + bDk3) - (bDk1 @ bDk3 - bDk3 @ bDk1 ) / 12
            bDk = bDk - omegaDkb + logMkb
            newError = np.linalg.norm(omegaDkb - logMkb)
            if newError > error:
                print(newError)
                bDk = bDk2
                break
            else:
                error = newError
                it += 1
        Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        for Rput, (Rorig, _) in self.ndegen.items():
            phase = self.bvec @ Rput
            rC[*Rput] = 1j *  np.einsum("xs,xmn,x->mns", ba, Rb[*Rorig], np.exp(-1j * np.pi * phase))
        return rC

    def calc_altLog(self):
        # Eq. (B1)
        nb = self.bvec.shape[0]
        mmnR = np.fft.fftn(self.M, axes=(0, 1, 2), norm="forward")
        s = self.M.shape
        Mrp = np.zeros((2*s[0], 2*s[1], 2*s[2], *s[3:]), dtype=complex)
        for Rput, (Rorig, nd) in self.ndegen.items():
            Mrp[*Rput] = mmnR[*Rorig] / nd
        mmnFine = np.fft.ifftn(Mrp, axes=(0, 1, 2), norm="forward")
        mmnShifted = np.zeros(s, dtype=complex)
        offsets = self.bvec * self.nkd
        for bi in range(nb):
            o = np.round(offsets[bi]).astype(int)
            mmnShifted[:, :, :, bi] = np.roll(mmnFine[:, :, :, bi], o, axis=(0, 1, 2))[::2,::2,::2]

        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                res = p.map(scipy.linalg.logm, [mmnShifted.reshape(self.nk, nb, self.nw, self.nw)[i//nb, i%nb] for i in range(nb*self.nk)])
        logMmn = np.array(res).reshape(s)
        rk = 1j * np.einsum("b,ba,xyzbmn->xyzmna", self.wb, self.k_crys2cart(self.bvec), logMmn)
        rr = np.fft.fftn(rk, axes=(0, 1, 2), norm="forward")
        rC = {}
        for Rput, (Rorig, _) in self.ndegen.items():
            rC[Rput] = rr[*Rorig]
        return rC

    def calc_altclog(self, maxIterations=20):
        # similar to Eqs. (41) - (45), but being based on Eq. (B1)
        nb = self.bvec.shape[0]
        nw = self.M.shape[-1]
        s = self.M.shape
        phaseFac = np.zeros((*s[0:3], nb), dtype=complex)
        for Rput, (Rorig, nc) in self.ndegen.items():
            phaseFac[Rorig] += np.exp(1j * np.pi * self.bvec @ Rput) / nc

        mmnR = np.fft.fftn(self.M, axes=(0, 1, 2), norm="forward")
        Mrp = np.zeros((2*s[0], 2*s[1], 2*s[2], *s[3:]), dtype=complex)
        for Rput, (Rorig, nd) in self.ndegen.items():
            Mrp[*Rput] = mmnR[*Rorig] / nd
        mmnFine = np.fft.ifftn(Mrp, axes=(0, 1, 2), norm="forward")
        mmnShifted = np.zeros(s, dtype=complex)
        offsets = self.bvec * self.nkd
        for bi in range(nb):
            o = np.round(offsets[bi]).astype(int)
            mmnShifted[:, :, :, bi] = np.roll(mmnFine[:, :, :, bi], o, axis=(0, 1, 2))[::2,::2,::2]
        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                res = p.map(scipy.linalg.logm, [mmnShifted.reshape(self.nk, nb, self.nw, self.nw)[i//nb, i%nb] for i in range(nb*self.nk)])
        logMmn = np.array(res).reshape(s)
        logMkb = logMmn
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        bLength = np.linalg.norm(ba, axis=1)
        bDk = logMkb
        error = np.linalg.norm(0.5*(bDk-np.swapaxes(bDk, -1, -2).conj())-logMkb)
        it = 0
        while it < maxIterations:
            print(f"\titeration {it}: error={error}")
            Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
            bDk1 = np.fft.ifftn(np.einsum("xyzbmn,xyzb->xyzbmn", Rb, phaseFac), axes=(0, 1, 2), norm="forward")
            bDk2 = bDk
            bDk3 = np.fft.ifftn(np.einsum("xyzbmn,xyzb->xyzbmn", Rb, np.conj(phaseFac)), axes=(0, 1, 2), norm="forward")
            # Magnus expansion of 4th order of path ordered integral
            omegaDkb = 1/6 * (bDk1 + 4 * bDk2 + bDk3) - (bDk1 @ bDk3 - bDk3 @ bDk1 ) / 12
            bDk = bDk - omegaDkb + logMkb
            newError = np.linalg.norm(omegaDkb - logMkb)
            if newError > error:
                print(newError)
                bDk = bDk2
                break
            else:
                error = newError
                it += 1
        Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        for Rput, (Rorig, _) in self.ndegen.items():
            rC[*Rput] = 1j * np.einsum("xs,xmn->mns", ba, Rb[*Rorig])
        return rC

    def calc_clog6(self, maxIterations=20):
        # Eqs. (41)-(45) however with 6th-order Magnus expansion
        nb = self.bvec.shape[0]
        nw = self.M.shape[-1]
        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                res = p.map(scipy.linalg.logm, [self.M.reshape(self.nk, nb, self.nw, self.nw)[i//nb, i%nb] for i in range(nb*self.nk)])
        logMkb = np.array(res).reshape(self.M.shape)
        bDk = logMkb
        error = np.linalg.norm(0.5*(bDk-np.swapaxes(bDk, -1, -2).conj())-logMkb)
        it = 0
        while it < maxIterations:
            print(f"\titeration {it}: error={error}")
            # Magnus expansion of 6th order of path ordered integral
            # (Blanes, Sergio, and Per Christian Moan.
            # "Fourth-and sixth-order commutator-free Magnus integrators for linear and non-linear dynamical systems." 
            # Applied Numerical Mathematics 56.12 (2006): 1519-1537.
            # Eq 18
            Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
            Ac = []
            for i in range(3):
                # compute derivatives of A at k+b/2 in b-direction
                derivFac = np.zeros((*logMkb.shape[0:3], nb), dtype=complex)
                for Rput, (Rorig, nc) in self.ndegen.items():
                    derivFac[Rorig] += (1j*np.pi*self.bvec @ Rput)**i / math.factorial(i) / nc
                Ac.append( np.fft.ifftn(np.einsum("xyzbmn,xyzb->xyzbmn", Rb, derivFac), axes=(0, 1, 2), norm="forward"))
            bDkOld = bDk
            cmt = lambda A,B : A @ B - B @ A
            NC = lambda *s: Ac[s[0]] if len(s)==1 else cmt(Ac[s[0]], NC(*s[1::])) # nested commutator
            omegaDkb = NC(0) + NC(2)/12 - NC(0,1)/12 + NC(1,2)/240 + NC(0,0,2)/360 - NC(1,0,1)/240 + NC(0,0,0,2)/720
            bDk = bDk - omegaDkb + logMkb
            newError = np.linalg.norm(omegaDkb - logMkb)
            if newError > error:
                print(newError)
                bDk = bDkOld
                break
            else:
                error = newError
                it += 1
        Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, self.k_crys2cart(self.bvec))
        for Rput, (Rorig, _) in self.ndegen.items():
            phase = self.bvec @ Rput
            rC[*Rput] = 1j *  np.einsum("xs,xmn,x->mns", ba, Rb[*Rorig], np.exp(-1j * np.pi * phase))
        return rC


def createParser():
    def schemeType(names=None):
        prefix = "calc_"
        schemes = [method[len(prefix):] for method in dir(WannierCalculator) 
                    if callable(getattr(WannierCalculator, method)) and method.startswith(prefix)]
        if names is None:
            return schemes
        schms = []
        for s in names.split(","):
            if not s in schemes:
                raise argparse.ArgumentTypeError(f"{s} is no supported interpolation scheme")
            schms.append(s)
        return schms

    parser = argparse.ArgumentParser(
                        description="""Calculates wannier matrix elements for multiple dipole interpolation schemes""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname (creates or reads seedname.npz)')
    parser.add_argument('-s', '--schemes', type=schemeType, 
                        help=f'comma separated list of interpolation schemes to evaluate (default: {",".join(schemeType())})',
                        default=schemeType() )
    parser.add_argument('-w', '--write_tb_dat', help='writes tight binding in human readable format', action='store_true')
    return parser


def main(seedname, schemes, write_tb_dat):
    fname = seedname + ".npz"
    if not os.path.exists(fname):
        print(f"Generate {fname}")
        np.savez_compressed(fname, **parse_all([seedname]))
    data = np.load(fname)
    wc = WannierCalculator(**data)

    realSpaceMats = { 'H' : wc.to_Mr(wc.H) }
    for k, vMat in wc.vMats.items():
        realSpaceMats[k] = wc.to_Mr(vMat)
    for scheme in schemes:
        print(f"Evaluating {scheme}-scheme")
        schemeFct = getattr(wc, "calc_" + scheme)
        rMat = schemeFct()
        if not rMat is None:
            realSpaceMats["R_" + scheme] = rMat

    if write_tb_dat:
        wc.write_tb_dat(seedname + "_custom_tb.dat", **realSpaceMats)
    np.savez_compressed(seedname + "_tb.npz", **wc.create_ksi_dict(**realSpaceMats))


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
