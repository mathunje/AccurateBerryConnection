#!/usr/bin/env python3

""" This scripts creates a wannier90 seedname_tb.dat compatible file from a seedname_tb.npz file.
"""

import argparse
from datetime import datetime
import numpy as np

import atu
from KspaceInterpolator import KspaceInterpolator

def extractBlocks(H, thrs=1e-10):
    maxS = -1
    for s in range(1, H.shape[0]):
        if np.allclose(H[s:, :s], 0, atol=thrs) and np.allclose(H[:s, s:],0, atol=thrs):
            maxS = s
    if maxS > 0:
        return maxS, *extractBlocks(H[maxS:, maxS:], thrs)
    return (H.shape[0], )


def extractBandExtrema(ksi, vbc):
    Nkd = max(ksi.rMats['H'].shape[0:3])
    H = ksi.k_grid("H", repeat=15 // Nkd +1)
    E = np.linalg.eigvalsh( H.reshape((-1, *H.shape[3::])) )
    vbMax = np.max(E[:, :vbc])
    cbMin = np.min(E[:, vbc:])
    return vbMax, cbMin


def main(seedname, scheme, gap_eV, vbc, outfname, nodat, nonpz):
    if seedname.endswith(".npz"):
        seedname = seedname[:-4]
    if seedname.endswith("_tb"):
        seedname = seedname[:-3]

    data = np.load(seedname + "_tb.npz")
    H = data['H']
    r0 = -data['RcellShift']
    lattice = data['lattice']
    schemeKey = scheme if scheme.startswith("R_") else "R_" + scheme
    if not schemeKey in data:
        rks = ",".join([ s[3:] for s in data if s.startswith("R_")])
        print(f"scheme {scheme} not in tb file. Available ones: {rks}")
    
    if outfname is None:
        fname_dat = f"{seedname}_{scheme}_tb.dat"
        fname_npz = f"{seedname}_{scheme}_tb.npz"
        if not (gap_eV is None):
            fname_dat = f"{seedname}_gap_{gap_eV:.3f}eV_{scheme}_tb.dat"
            fname_npz = f"{seedname}_gap_{gap_eV:.3f}eV_{scheme}_tb.npz"
    else:
        fname_dat = outfname + "_tb.dat" 
        fname_npz = outfname + "_tb.npz"

    Ar = data[schemeKey]
    Hr = data['H']
    HrSet = np.linalg.norm(Hr, axis=(3, 4)) > 1e-10
    ArSet = np.sum(np.linalg.norm(Ar, axis=(3,4)), axis=3) > 1e-10
    entriesSet = HrSet | ArSet
    if not (gap_eV is None):
        new_gap = atu.from_eV(float(gap_eV))
        ksi = KspaceInterpolator(lattice, data['RcellShift'], H=Hr, R=Ar)
        if vbc is None:
            nws = extractBlocks(H[*r0])
            print(f"found blocks of sizes {nws}")
            assert len(nws) == 2
            vbc = nws[0]
        vbMax, cbMin = extractBandExtrema(ksi, vbc)
        print(f"Current band gap {atu.to_eV(cbMin-vbMax):.3f} eV")
        cbShift = new_gap - cbMin + vbMax
        dictK = ksi.k_grid()
        Hk = dictK['H']
        Ak = dictK['R']
        origShape = Hk.shape
        Nw = origShape[-1]
        Hk = Hk.reshape((-1, Nw, Nw))
        Ak = Ak.reshape((-1, Nw, Nw, 3))
        Ek, Uk = np.linalg.eigh(Hk)
        Ek_new = Ek.copy()
        Ek_new[:, vbc:] += cbShift
        # scale berry connection to be consistent with energy shift
        AH = np.einsum("sba,sbcx,scd->sadx", Uk.conj(), Ak, Uk)
        AH[:, :vbc, vbc:] *= (Ek[:,:vbc, None, None] - Ek[:,None,vbc:, None]) / (Ek_new[:,:vbc,None, None] - Ek_new[:,None,vbc:, None]) 
        AH[:, vbc:, :vbc] *= (Ek[:,None, :vbc, None] - Ek[:,vbc:,None, None]) / (Ek_new[:,None,:vbc, None] - Ek_new[:,vbc:,None, None]) 
        Ak = np.einsum("sab,sbcx,sdc->sadx", Uk, AH, Uk.conj())
        Ar = np.fft.fftn(Ak.reshape((*origShape, 3)), axes=(0, 1, 2), norm='forward')
        Ar = np.roll(Ar, r0, axis=(0, 1, 2))
        Hr[*r0][vbc:, vbc:] += np.identity(H[*r0].shape[0] -vbc) * cbShift

        Ar[~entriesSet] = 0
        if not nonpz:
            np.savez_compressed(fname_npz,  **{ 'RcellShift' : data['RcellShift'], 
                                                'lattice' : data['lattice'],
                                                'H' : Hr, schemeKey : Ar} )
    if not nodat:
        with open(fname_dat, "w") as f:
            header = datetime.now().strftime("%Y/%m/%d, %H:%M:%S") + " " + scheme
            if not (gap_eV is None):
                header += f" with shifted band gap from {atu.to_eV(cbMin-vbMax):.3f}eV to {gap_eV:.3f}eV"
            f.write(header + "\n")
            for i in range(3):
                f.write(" " + "    ".join([f"{atu.to_A(lattice[i, j]):+.15f}" for j in range(3)]) + "\n")
            f.write(f"{Hr.shape[-1]:8d}\n")
            cnt = np.sum(entriesSet)
            f.write(f"{cnt:8d}\n")
            while cnt > 0:
                f.write("".join([f"{s:>5d}" for s in (1, ) * min(cnt, 15) ]) + "\n")
                cnt -= 15
            for M, scale in [(Hr, atu.to_eV(1)), (Ar, atu.to_A(1))]:
                for rc in zip(*[ s[entriesSet] for s in np.indices((Hr.shape[0:3]))]):
                    f.write("\n")
                    f.write("".join([f"{s-c:>5d}" for s,c in zip(rc, r0)]) + "\n")
                    Mr = M[*rc]
                    for i in range(Mr.shape[0]):
                        for j in range(Mr.shape[1]):
                            entries = Mr[j, i].flatten().view(float)
                            f.write(f"{j+1:>5d}{i+1:>5d}  " + " ".join([f"{scale*e:+.12e}" for e in entries]) + "\n")


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Calculates wannier matrix elements for multiple dipole interpolation schemes""",
                        epilog="from MT")
    parser.add_argument('seedname', help='seedname (reads seedname.npz)')
    parser.add_argument('-s', '--scheme',
                        help=f'interpolation scheme used for output of position operator matrix',
                        default='clog')
    parser.add_argument('-g', '--gap', help='adapts band gap to given value [eV] by shifting the conduction bands',
                        dest='gap_eV', type=float)
    parser.add_argument('-o', '--output', help='prefix of output file name (_tb.dat or _tb.npz is added automatically)', dest='outfname')
    parser.add_argument('-v', '--valenceBandCount', help='number of valence bands', type=int, dest='vbc')
    parser.add_argument('-nw', '--nodat', help='disable writing of seedname_tb.dat', action='store_true')
    parser.add_argument('-np', '--nonpz', help='disable writing of seedname_tb.npz', action='store_true')
    return parser


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
