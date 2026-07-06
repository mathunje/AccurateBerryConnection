#!/usr/bin/env python3

""" Parses all required files for the calculation of the Berry connection.
The velocity matrix file is optional and enables quantitative comparisson
of the different schemes.
"""

import numpy as np
import os
import re
import sys

import atu

def _parseToNp(dtype, s):
    return np.array([dtype(v) for v in s.split()])

def _parseToCmplx(s):
    r, i = [float(v) for v in s.split()]
    return r + 1j*i


def parse_u_mat(fname):
    with open(fname, "r") as f:
        f.readline() # header
        Nk, Nw, Nb = _parseToNp(int, f.readline())
        Udis = np.empty((Nk, Nw, Nb), dtype=complex)
        kvec = np.empty((Nk, 3), dtype=float)
        for n in range(Nk):
            f.readline() # empty line
            kvec[n] = _parseToNp(float, f.readline())
            for i in range(Nw):
                for j in range(Nb):
                    Udis[n, i, j] = _parseToCmplx(f.readline())
    # restore unitarity as best as possible (reduce truncation error of input)
    u, s, vh = np.linalg.svd(Udis, full_matrices=False)
    return kvec, np.swapaxes(u @ vh, 1, 2)

def parse_bvec(fname):
    with open(fname, "r") as f:
        f.readline() # header
        Nk, Nw = _parseToNp(int, f.readline())
        bvec = np.empty((Nk, Nw, 3), dtype=float)
        weight = np.empty((Nk, Nw), dtype=float)
        for n in range(Nk):
            for i in range(Nw):
                v = _parseToNp(float, f.readline())
                bvec[n, i] = v[0:3]
                weight[n, i] = v[3]
    return bvec, weight

def parse_v(fname):
    data = np.loadtxt(fname).T
    Nb, _, Nk = [int(np.max(data[i])) for i in range(3)]
    v = data[3::].T.reshape((Nk, Nb, Nb, 6)).swapaxes(1, 2)
    v = v[...,0::2] + 1j * v[...,1::2]
    return v

def parse_mmn(fname):
    with open(fname, 'r') as f:
        f.readline() # header
        num_bands, num_kpts, nntot = _parseToNp(int, f.readline())
        kIndex = np.empty((num_kpts, nntot), dtype=int)
        M = np.empty( (num_kpts, nntot, num_bands, num_bands), dtype=complex)
        G = np.empty( (num_kpts, nntot, 3), dtype=int)
        for kptf in range(num_kpts):
            for n in range(nntot):
                ka, kb, G0, G1, G2 = [int(s) for s in f.readline().split()]
                # offset of 1 as reference indices are counted starting from 1 instead of 0
                assert ka == kptf + 1
                kIndex[kptf, n] = kb - 1
                M[kptf, n] = np.array([_parseToCmplx(f.readline()) for _ in range(num_bands**2)]).reshape((num_bands, num_bands)).T
                G[kptf, n] = np.array([G0, G1, G2])
    return M, kIndex, G

def parse_nnkp_lattice_rguide(fname):
    lines = open(fname, 'r').readlines()
    def findBlock(name):
        for start, l in enumerate(lines):
            if l.strip() == f"begin {name}":
                break
        start += 1
        if start == len(lines):
            return -1, -1
        for end in range(start, len(lines)):
            if lines[end].strip() == f"end {name}":
                break
        return start, end
    start, end = findBlock("real_lattice")
    lattice = np.array([_parseToNp(float, l.strip()) for l in lines[start:end]])
    start, end = findBlock("projections")
    if start < 0:
        return lattice, np.empty([0, 3])
    return lattice, np.array([_parseToNp(float, l.strip())[:3] for l in lines[start+1:end:2]])

def parse_select_projections(fname, nProj):
    lines = open(fname, 'r').readlines()
    lines = [ l.split("#", 1)[0] for l in  lines]
    spKey = "select_projections"
    sp = [ l.partition(spKey)[2].strip() for l in lines if spKey in l][0]
    if sp[0] == "=":
        sp = sp[1:]
    proj = np.zeros(nProj, dtype=bool)
    for subStr in re.split(",| ", sp):
        if len(subStr) == 0:
            continue
        if "-" in subStr:
            b, e = [int(i) for i in subStr.split("-")]
            proj[b-1:e] = True
        else:
            proj[int(subStr)-1] = True
    return proj


def parse_eig(fname):
    n, kpt, E = np.loadtxt(fname).T
    E = E.reshape( (int(np.max(kpt)), int(np.max(n))) )
    return E


def parse_all(seednames):
    for i, seedname in enumerate(seednames):
        if i == 0:
            kpts, U = parse_u_mat(seedname + "_u.mat")
            E = parse_eig(seedname + ".eig")
            M, kIndex, G = parse_mmn(seedname + ".mmn")
            lattice, rguide = parse_nnkp_lattice_rguide(seedname + ".nnkp")
            selectedProj = parse_select_projections(seedname + ".win", rguide.shape[0])
            if os.path.exists(seedname + "_u_dis.mat"):
                kvec, Udis = parse_u_mat(seedname + "_u_dis.mat")
                uMat = Udis @ U
            else:
                uMat = U
            continue
        cE = parse_eig(seedname + ".eig")
        assert np.allclose(E, cE)
        cM, ckIndex, cG = parse_mmn(seedname + ".mmn")
        assert np.allclose(cM, M)
        assert np.allclose(ckIndex, kIndex)
        assert np.allclose(cG, G)
        clattice, _ = parse_nnkp_lattice_rguide(seedname + ".nnkp")
        assert np.allclose(clattice, lattice)
        ckpts, cU = parse_u_mat(seedname + "_u.mat")
        if os.path.exists(seedname + "_u_dis.mat"):
            ckvec, cUdis = parse_u_mat(seedname + "_u_dis.mat")
            assert np.linalg.norm( np.einsum("kab,kac->kbc", Udis, np.conj(cUdis)) ) < 1e-10
            cuMat = cUdis @ cU
        else:
            cuMat = cU
        uMat = np.concatenate( (uMat, cuMat), axis=2)
    # check that MP-grid (without shift)
    kUnique = [ np.unique(kpts[:, i]) for i in range(kpts.shape[1])]
    nkd = []
    for i in range(3):
        r = np.unique(kpts[:, i])
        assert np.allclose(r, np.linspace(0, 1, r.size, endpoint=False))
        nkd.append(r.size)
    nkd = np.array(nkd)
    assert np.prod(nkd) == kpts.shape[0]

    Mw = np.zeros((*M.shape[:-2], uMat.shape[-1], uMat.shape[-1]), dtype=complex)
    for k, kbs in enumerate(kIndex):
        for ind, kb in enumerate(kbs):
            Mw[k, ind] = uMat[k].conj().T @ M[k, ind] @ uMat[kb]
    H = np.einsum("kba,kb,kbc->kac", uMat.conj(), E, uMat)

    # shape assertion for FFT
    # can be made more flexible by implementing resorting
    expected = np.mgrid[0:nkd[0], 0:nkd[1], 0:nkd[2]]
    for i in range(3):
        assert np.allclose((expected[i] / nkd[i]).flatten(), kpts[:, i])

    # sort neighbour indices to get common bVec order for all k-points
    for k, kbs in enumerate(kIndex):
        kDiff = np.array([kpts[kb]-kpts[k] for kb in kbs]) + G[k]
        if k == 0:
            bVec = kDiff
            continue
        aligned_bvec_order = []
        for si in range(kDiff.shape[0]):
            ib = 0
            while ib < bVec.shape[0]:
                if np.allclose(kDiff[si], bVec[ib]):
                    break
                ib += 1
            assert ib < bVec.shape[0]
            aligned_bvec_order.append(ib)
        inverse_order = np.argsort(aligned_bvec_order)
        Mw[k] = Mw[k, inverse_order]
        kIndex[k] = kIndex[k, inverse_order]
    res = { 'bvec' : bVec, 
            'lattice' : atu.from_A(lattice),
            'rguide' : rguide[selectedProj],
            'H' : atu.from_eV(H.reshape((*nkd, *H.shape[1::])) ), 
            'M' : Mw.reshape( (*nkd, *Mw.shape[1::]) ) }
    for vn in ['v', 'p']:
        fname = seedname + "." + vn
        if os.path.exists(fname):
            v = atu.from_eV(parse_v(fname)) * atu.from_A(1)
            res[vn] = np.einsum("kba,kbcs,kcd->kads", uMat.conj(), v, uMat).reshape((*nkd, *H.shape[1::], 3))
    return res


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} seedname [outfile.npz]")
        print(f"       {sys.argv[0]} seedname1 [seedname2] ... seednameN outfile.npz")
        print(f"   The second option merges the different Wannierizations together and works\n"
              f"   only for the development version of wannier90 (e.g. commit f5ba0a9), see issue 608")
        exit(0)
    if len(sys.argv) == 2:
        seedname = sys.argv[1]
        gridDict = parse_all([seedname])
        np.savez_compressed(seedname + ".npz", **gridDict)
        exit(0)
    else:
        gridDict = parse_all(sys.argv[1:-1:])
        np.savez_compressed(sys.argv[-1] + ".npz", **gridDict)

