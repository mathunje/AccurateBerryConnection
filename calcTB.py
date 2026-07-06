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


from scipy.linalg._matfuncs_inv_ssq import _logm_triu as logm_triu
from threadpoolctl import threadpool_limits

import atu
from inputParser import parse_all

import itertools
from scipy.cluster.hierarchy import DisjointSet


def branchedLogm(A, brguide, blockEps=1e-3):
    """ Computes matrix logarithm with guiding centers as described in paper.
        Details from doi.org/10.1137/S0895479802410815
    """
    T, U = scipy.linalg.schur(A)
    T_ii = np.diag(T)
    blocks = DisjointSet(list(range(T_ii.size)))
    for a in range(T_ii.size):
        for b in range(a+1, T_ii.size):
            if abs(T_ii[a] - T_ii[b]) < blockEps:
                blocks.merge(a, b)
    bindex = np.zeros(T_ii.shape)
    for i, block in enumerate(blocks.subsets()):
        bindex[list(block)] = i
    # insertion sort of blocks
    for i in range(1, T_ii.size):
        j = i
        while j > 0 and bindex[j-1] > bindex[j]:
            # move eigenvalue inplace (fortran indexing)
            scipy.linalg.lapack.ztrexc(T, U, j, j+1, 1, 1, 1)
            bindex[[j-1, j]] = bindex[[j, j-1]]
            j -= 1
    # apply logm on diagonal containing blocks with intentionally selected branch
    logT = np.zeros(T.shape, dtype=complex)
    guidingRef = np.einsum("ba,b,ba->a", np.conj(U), -2j*np.pi * brguide, U)
    start = 0
    for cnt in np.unique(bindex, return_counts=True)[1]:
        end = start + cnt
        logT[start:end, start:end] = logm_triu(T[start:end, start:end])
        off = np.imag(guidingRef[start:end] - np.diag(logT)[start:end] ) / (2*np.pi)
        round_off = np.round(off)
        if not np.allclose(round_off, round_off[0]):
            # different branches are only expected for phases +/- pi thus rotate matrix by pi to force same branches
            logT[start:end, start:end] = logm_triu(-T[start:end,start:end]) + 1j*np.pi * np.identity(end-start)
            off = np.imag(guidingRef[start:end] - np.diag(logT)[start:end] ) / (2*np.pi)
            round_off = np.round(off)
            assert np.allclose(round_off, round_off[0])
        logT[start:end, start:end] += np.diag(round_off) * 2j*np.pi
        start = end
    # fill remaining entries
    for i in reversed(range(T_ii.size)):
        for j in range(i+1, T_ii.size):
            if blocks.connected(i, j):
                continue
            en = T[i,j] * (logT[i, i] - logT[j,j])
            if j - i >= 2:
                en += np.sum( logT[i, i+1:j] * T[i+1:j,j] - T[i,i+1:j]* logT[i+1:j,j] )
            logT[i, j] = en / (T[i, i] - T[j, j])
    # for testing uncomment: 
    # print(np.linalg.norm(scipy.linalg.expm(logT)-T))
    return np.einsum("ab,bc,dc->ad", U, logT, np.conj(U))


def branchedLogmBatched(Abatch, brguide):
    res = np.empty(Abatch.shape, dtype=complex)
    #for i in range(Abatch.shape[0]):
    #    res[i] = branchedLogm(Abatch[i], brguide[:, i])
    for i in range(Abatch.shape[0]):
        for j in range(Abatch.shape[1]):
            res[i, j] = branchedLogm(Abatch[i, j], brguide[:, j])
    return res


def logM(M, brguide):
    coreCount = psutil.cpu_count(logical=False)
    # res = [branchedLogmBatched(s, brguide) for s in M.reshape((-1, *M.shape[-3:]))]
    with threadpool_limits(limits=1, user_api='openmp'):
        with Pool(coreCount) as p:
            res = p.starmap(branchedLogmBatched, [ (s, brguide) for s in np.array_split(M.reshape((-1, *M.shape[-3:])),
                                                                                        coreCount)])
    return np.concatenate(res, axis=0).reshape(M.shape)


class WannierCalculator:
    def __init__(self, H, M, lattice, rguide, bvec, **vMats):
        self.H = H
        self.M = M
        self.lattice = lattice
        self.rguide = rguide
        self.bvec = bvec
        self.vMats = vMats
        self.recipLattice = 2*np.pi * np.linalg.inv(self.lattice).T
        self.nkd = self.M.shape[0:3]
        self.nk = np.prod(self.nkd)
        self.nw = self.M.shape[-1]
        self.ndegen = self.calcNdegen()
        self.wb = self.calcWeights(self.bvec)
        self.brguide = self.rguide @ self.bvec.T

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
    # The equation numbers refer to the manuscript 
    # "Self-consistent evaluation of the Berry connection for Wannier functions"
    # arxiv.org/abs/2604.21660
    ##############################################


    def guided_brk_MV(self, Mii):
        brk = np.log(Mii)
        off = np.round(brk.imag / (2*np.pi) - self.brguide.T)
        return brk - 2j * np.pi * off

    def guided_r_SS(self, Mii):
        br = np.log(np.sum(Mii, axis=(0,1,2)) / self.nk)
        off = np.round(br.imag / (2*np.pi) - self.brguide.T)
        br -= 2j * np.pi * off
        return - np.einsum("b,ba,bm->ma", self.wb, self.k_crys2cart(self.bvec), br.imag)

    def calc_MV(self):
        Mmod = self.M.copy()
        # Eq. (25)
        np.einsum("xyzsaa->xyzsa", Mmod)[:] = self.guided_brk_MV(np.einsum("xyzsaa->xyzsa", self.M))
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
        np.einsum("xyzsaa->xyzsa", Mmod)[:] = self.guided_brk_MV(np.einsum("xyzsaa->xyzsa", self.M))
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
        R0 = self.guided_r_SS(np.einsum("xyzsaa->xyzsa", self.M))
        # refinement according to eq.47 of arxiv.org/pdf/2604.22614
        bvec_cart = self.k_crys2cart(self.bvec)
        mkSum = np.einsum("baa->ba", np.sum(self.M, axis=(0,1,2)))
        #for i in range(3):
        #    R0 -= np.einsum("b,ba,bm->ma", self.wb, bvec_cart, np.exp(1j * bvec_cart @ R0.T) *mkSum ).imag / self.nk
        mmnR = np.fft.fftn(self.M, axes=(0, 1, 2), norm="forward")
        rC = {}
        ba = np.einsum("a,ab->ab", self.wb, bvec_cart)
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
        
        logMmn = logM(self.M, self.brguide)
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
        logMmn = logM(self.M, self.brguide)
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

    def calc_clogFull(self, maxIterations=20):
        # Eqs. (41) - (45)
        nb = self.bvec.shape[0]
        nw = self.M.shape[-1]
        bvec_cart = self.k_crys2cart(self.bvec)
        ba = np.einsum("a,ab->ab", self.wb, bvec_cart)
        logMkb = logM(self.M, self.brguide)

        logMkb_cmp = 0.5*(logMkb-np.swapaxes(logMkb, -1, -2).conj())

        phaseFac = np.zeros((*logMkb.shape[0:3], nb), dtype=complex)
        phaseFac2 = np.zeros((*logMkb.shape[0:3], nb), dtype=complex)
        for Rput, (Rorig, nc) in self.ndegen.items():
            phaseFac[Rorig] += np.exp(-1j * np.pi * self.bvec @ Rput) / nc
            phaseFac2[Rorig] += np.exp(2j * np.pi * self.bvec @ Rput) / nc

        def evalMagnus(AR):
            ARb = np.einsum("abcsmn,xs->abcxmn", AR, bvec_cart)
            # 1: bA^k | 2: bA^{k+b/2} | 3: bA^{k+b}
            bA1 = np.fft.ifftn(ARb, axes=(0, 1, 2), norm="forward")
            bA2 = np.fft.ifftn(np.einsum("abcxmn,abcx->abcxmn", ARb, np.conj(phaseFac)), axes=(0, 1, 2), norm="forward")
            bA3 = np.fft.ifftn(np.einsum("abcxmn,abcx->abcxmn", ARb, phaseFac2), axes=(0, 1, 2), norm="forward")
            return (bA1 + 4 * bA2 + bA3) / 6 - (bA1 @ bA3 - bA3 @ bA1 ) / 12

        bDk = logMkb
        AR = np.einsum("xs,abcxmn,abcx->abcsmn", ba, np.fft.fftn(bDk, axes=(0, 1, 2), norm='forward'), phaseFac)
        Ikb = evalMagnus(AR)
        error = np.linalg.norm(Ikb-logMkb_cmp)
        it = 0
        while it < maxIterations:
            print(f"\titeration {it}: error={error}")
            bDk_new = bDk - Ikb + logMkb
            AR = np.einsum("xs,abcxmn,abcx->abcsmn", ba, np.fft.fftn(bDk_new, axes=(0, 1, 2), norm='forward'), phaseFac)
            Ikb = evalMagnus(AR)
            newError = np.linalg.norm(Ikb - logMkb_cmp)
            if newError > error:
                print(f"Break as increased error was observed: {newError}")
                bDk = bDk_new
                break
            else:
                error = newError
                bDk = bDk_new
                it += 1
        Rb = np.fft.fftn(bDk, axes=(0, 1, 2), norm="forward")
        rC = {}
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
        logMmn = logM(mmnShifted, self.brguide)
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
        logMmn = logM(mmnShifted, self.brguide)
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
        logMkb = logM(self.M, self.brguide)
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
