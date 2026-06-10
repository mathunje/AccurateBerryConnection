#!/usr/bin/env python3

import numpy as np
import sys
import os
from datetime import datetime
import re

def toCmplx(s):
    a, b = [float(v) for v in s.split()]
    return a + 1j * b

def execW90chk2chk(seed, mode):
    head, tail = os.path.split(seed)
    if head != "":
        oldDir = os.getcwd()
        os.chdir(head)
    os.system(f"w90chk2chk.x -{mode} {tail}")
    if head != "":
        os.chdir(oldDir)

def loadCheckpoint(seed):
    if not os.path.exists(seed + ".chk.fmt"):
        if not os.path.exists(seed + ".chk"):
            print(f"loadCheckpoint: no checkpoint found for {seed}")
        else:
            execW90chk2chk(seed, "export")
    else:
        if os.path.exists(seed + ".chk"):
            chkTime = os.path.getmtime(seed + ".chk")
            fmtTime = os.path.getmtime(seed + ".chk.fmt")
            if chkTime > fmtTime:
                execW90chk2chk(seed, "export")

    f = open(seed + ".chk.fmt") 
    d = {}
    d['header'] = f.readline()[:-1]
    num_bands = int(f.readline())
    d['num_bands'] = num_bands
    num_exclude_bands = int(f.readline())
    eb = np.empty((num_exclude_bands), dtype=np.int32)
    for i in range(num_exclude_bands):
       eb[i] = int(f.readline())
    d['excluded_bands'] = eb
    d['realLattice'] = np.array([float(s) for s in f.readline().split()]).reshape((3, 3)).T
    d['recipLattice'] = np.array([float(s) for s in f.readline().split()]).reshape((3, 3)).T
    num_kpts = int(f.readline())
    d['mp_grid'] = np.array([int(s) for s in f.readline().split()])
    kpts = np.empty((num_kpts, 3))
    for i in range(num_kpts):
        kpts[i, :] = np.array([float(s) for s in f.readline().split()])
    d['kpts'] = kpts
    nntot = int(f.readline())
    num_wann = int(f.readline())
    d['num_wann'] = num_wann
    d['checkpoint'] = f.readline()[:-1]
    disentangled = bool(int(f.readline()))
    d['disentangled'] = disentangled
    if disentangled:
        d['omega_invariant'] = float(f.readline())
        d['lwindow'] = np.array([ bool(int(f.readline())) for _ in range(num_bands * num_kpts)]).reshape(num_kpts, num_bands)
        d['ndimwin'] = np.array([ int(f.readline()) for _ in range(num_kpts)])
        d['u_matrix_opt'] = np.array([toCmplx(f.readline()) for _ in range(num_bands * num_wann * num_kpts)]).reshape(
                                        (num_kpts, num_wann, num_bands))
    else:
        d['omega_invariant'] = 0
        d['lwindow'] = np.ones((num_kpts, num_bands), dtype=bool)
        d['ndimwin'] = num_bands * np.ones((num_kpts), dtype=int)
        d['u_matrix_opt'] = np.empty((num_kpts, num_wann, num_bands), dtype=complex)
        for i in range(num_kpts):
            d['u_matrix_opt'][i, :, :] = np.identity(num_wann)

    d['u_matrix'] = np.array([toCmplx(f.readline()) for _ in range(num_wann**2 * num_kpts)]).reshape(
                                    (num_kpts, num_wann, num_wann))
    d['m_matrix'] = np.array([toCmplx(f.readline()) for _ in range(num_wann**2 * nntot * num_kpts)]).reshape(
                                    (num_kpts, nntot, num_wann, num_wann))
    wannier_centres = np.empty((num_wann, 3))
    for i in range(num_wann):
        wannier_centres[i, :] = np.array([float(s) for s in f.readline().split()])
    d['wannier_centres'] = wannier_centres
    d['wannier_spreads'] = np.array([float(f.readline()) for _ in range(num_wann)])
    f.close()
    return d


def writeCheckpoint(d, seed):
    f = open(seed + ".chk.fmt", 'w')
    f.write(f"{d['header']}\n")
    f.write(f"{d['num_bands']}\n")
    eb = d['excluded_bands']
    f.write(f"{eb.size}\n")
    for e in eb:
        f.write(f"{e}\n")
    f.write("".join([f"{s:>25.17g}" for s in d['realLattice'].T.flatten()]) + "\n")
    f.write("".join([f"{s:>25.17g}" for s in d['recipLattice'].T.flatten()]) + "\n")
    kpts = d['kpts']
    f.write(f"{kpts.size // 3}\n")
    f.write(" ".join([str(nd) for nd in d['mp_grid']]) + "\n")
    for kpt in kpts:
        f.write("".join([f"{s:>25.17g}" for s in kpt]) + "\n")
    f.write(f"{d['m_matrix'].shape[1]}\n")
    f.write(f"{d['num_wann']}\n")
    f.write(d['checkpoint'] + "\n")
    f.write(f"{int(d['disentangled'])}\n")
    if d['disentangled']:
        f.write(f"{d['omega_invariant']:>25.17g}\n")
        f.write("\n".join([str(int(v)) for v in d['lwindow'].flatten()]) + "\n")
        f.write("\n".join([str(v) for v in d['ndimwin']]) + "\n")
        f.write("\n".join([f"{v.real:>25.17g}{v.imag:>25.17g}" for v in d['u_matrix_opt'].flatten()]) + "\n")
    f.write("\n".join([f"{v.real:>25.17g}{v.imag:>25.17g}" for v in d['u_matrix'].flatten()]) + "\n")
    f.write("\n".join([f"{v.real:>25.17g}{v.imag:>25.17g}" for v in d['m_matrix'].flatten()]) + "\n")
    for wc in d['wannier_centres']:
        f.write("".join([f"{s:>25.17g}" for s in wc]) + "\n")
    f.write("\n".join([str(v) for v in d['wannier_spreads']]) + "\n")
    f.close()
    execW90chk2chk(seed, "import")


def load_m_matrix(seed):
    fname = seed + ".mmn"
    if not os.path.exists(fname):
        print(f"load_m_matrix: {fname} does not exists")
        return
    f = open(fname, 'r')
    f.readline() # header
    num_bands, num_kpts, nntot = [int(s) for s in f.readline().split()]
    kIndex = np.empty((num_kpts, nntot), dtype=int)
    m = np.empty( (num_kpts, nntot, num_bands, num_bands), dtype=np.complex128)
    for kptf in range(num_kpts):
        for n in range(nntot):
            ka, kb, _, _, _ = [int(s) for s in f.readline().split()] # ignore G vector
            # offset of 1 as reference indices are counted starting from 1 instead of 0
            assert ka == kptf + 1
            kIndex[kptf, n] = kb - 1 
            m[kptf, n] = np.array([toCmplx(f.readline()) for _ in range(num_bands**2)]).reshape((num_bands, num_bands))
    return m, kIndex


def applyUnitary(m, u, kIndex):
    res = np.empty((m.shape[0], m.shape[1], u.shape[1], u.shape[1]), dtype=np.complex128)
    for k, kbs in enumerate(kIndex):
        for ind, kb in enumerate(kbs):
            # inverse multiplication order as the matrices are read in transposed order
            res[k, ind] = u[kb] @ m[k, ind] @ u[k].conj().T
    return res


def mergeCheckpoints(dcb, dvb, m_data):
    m, kIndex = m_data
    dres = {}
    dres['header'] = "Cb/Vb: " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    # bandsInCalc = dcb['num_bands'] + dcb['excluded_bands'].size
    # num_bands = bandsInCalc - dvb['excluded_bands'].size
    num_bands = dcb['num_bands'] + dvb['num_bands']
    dres['num_bands'] = num_bands
    num_kpts = dcb['kpts'].size // 3
    for copyData in ['excluded_bands', 'realLattice', 'recipLattice', 'mp_grid', 'kpts']:
        dres[copyData] = dvb[copyData]
    for copyData in ['checkpoint', 'disentangled']: 
        dres[copyData] = dcb[copyData]
    dres['omega_invariant'] = dvb['omega_invariant'] + dcb['omega_invariant']
    num_wann = dcb['num_wann'] + dvb['num_wann']
    dres['num_wann'] = num_wann
    dres['wannier_centres'] = np.concatenate((dvb['wannier_centres'], dcb['wannier_centres']))
    dres['wannier_spreads'] = np.concatenate((dvb['wannier_spreads'], dcb['wannier_spreads']))
    if dcb['disentangled'] + dvb['disentangled'] == 0:
        dres['excluded_bands'] = np.intersect1d(dvb['excluded_bands'], dcb['excluded_bands'])
    elif dcb['disentangled'] + dvb['disentangled'] == 1:
        dres['lwindow'] = np.concatenate((dvb['lwindow'], dcb['lwindow']), axis=1)
        dres['ndimwin'] = dcb['ndimwin'] + dvb['ndimwin']
        u_matrix_opt = np.zeros( (num_kpts, num_wann, num_bands), dtype=np.complex128)
        u_matrix_opt[:, :dvb['num_wann'], :dvb['num_bands']] = dvb['u_matrix_opt']
        u_matrix_opt[:, dvb['num_wann']::,dvb['num_bands']::] = dcb['u_matrix_opt']
        dres['u_matrix_opt'] = u_matrix_opt
        m = applyUnitary(m, u_matrix_opt, kIndex)
    elif dcb['disentangled'] + dvb['disentangled'] == 2:
        assert dcb['num_bands'] == dvb['num_bands']
        assert not np.any( dvb['lwindow'] & dcb['lwindow'])
        num_bands = dcb['num_bands']
        dres['num_bands'] = num_bands
        dres['lwindow'] = dvb['lwindow'] | dcb['lwindow']
        dres['ndimwin'] = dcb['ndimwin'] + dvb['ndimwin']
        u_matrix_opt = np.zeros( (num_kpts, num_wann, num_bands), dtype=np.complex128)
        for k in range(num_kpts):
            lv = dvb['lwindow'][k]
            ndv = dvb['ndimwin'][k] # = np.sum(lv)
            lc = dcb['lwindow'][k]
            ndc = dcb['ndimwin'][k] # = np.sum(lc)
            p = (1 * lv + 2 * lc)
            p = p[p>0]
            p1 = np.argwhere(p==1).flatten()
            p2 = np.argwhere(p==2).flatten()
            u_matrix_opt[k, :dvb['num_wann'], p1] = dvb['u_matrix_opt'][k][:, :ndv]
            u_matrix_opt[k, dvb['num_wann']::][:, p2] = dcb['u_matrix_opt'][k][:, :ndc]
        dres['u_matrix_opt'] = u_matrix_opt
        m = applyUnitary(m, u_matrix_opt, kIndex)
    u_matrix = np.zeros((num_kpts, num_wann, num_wann), dtype=np.complex128)
    u_matrix[:, :dvb['num_wann'], :dvb['num_wann']] = dvb['u_matrix']
    u_matrix[:, dvb['num_wann']::, dvb['num_wann']::] = dcb['u_matrix']
    dres['u_matrix'] = u_matrix
    dres['m_matrix'] = applyUnitary(m, u_matrix, kIndex)
    return dres


def main(baseDir, seed):
    subDirs = [d for d in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, d)) ]
    fullDir = os.path.join(baseDir, "cb-vb")
    cbDir = os.path.join(baseDir, "cb")
    vbDir = os.path.join(baseDir, "vb")
    print(f"Merging {vbDir}, {cbDir} -> {fullDir}")
    if not os.path.exists(cbDir):
        print(f"{cbDir} does not exist")
        return
    if not os.path.exists(vbDir):
        print(f"{cbDir} does not exist")
        return
    dcb = loadCheckpoint(os.path.join(cbDir, seed))
    dvb = loadCheckpoint(os.path.join(vbDir, seed))
    mergeSeed = os.path.join(fullDir, seed)
    m_data = load_m_matrix(mergeSeed)
    dm = mergeCheckpoints(dcb, dvb, m_data)
    writeCheckpoint(dm, mergeSeed)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Merges to checkpoint files from valence and conduction band wannierization (v3.1) into a single one")
        print("Seaches for cb and vb subdirectories in baseDir and stores the resuilt in baseDir/cb-vb")
        print("Requires wannier90.x and w90chk2chk.x")
        print(f"Usage: {sys.argv[0]} baseDir seedname")
    else:
        main(sys.argv[1], sys.argv[2])
