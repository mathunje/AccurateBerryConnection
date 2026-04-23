### Accurate Berry connection for solids

This repository provides the implementation of different interpolation schemes for the Berry connection in solids within the Wannier90 framework.
The schemes itself are described in the following publication:

Martin Thümmler, Thomas Lettau, Alexander Croy, Ulf Peschel, Stefanie Gräfe,
*Self-consistent evaluation of the Berry connection for Wannier functions*, ...

We kindly ask you to cite it if you use our provided scripts or the ideas therein.

## Patch of Quantum Espresso
To obtain the velocity matrix elements in Quantum Espresso (7.4) we patched the *pw2wannier90* program and provide the patch in the *patch* directory.
It can not handle symmetries. The output format of the velocity matrix elements is inspired by the seedname.amn format of Wannier90.

## Requirements
The scripts were developed using Python 3.
- numpy 2.2.3
- scipy 1.15.2
- matplotlib 3.10.1
- psutil 7.0.0
- threadpoolctl 3.5.0
You can also load a development shell using *nix*.

## Reproduction
The directory *inputs* contains a zip with all input files for Quantum Espresso and wannier90 (3.1).

## Calculation steps
- Run a DFT calculation. For QE you can use the patched version of *pw2wannier90* with *write_v* and *write_p* in the input file to obtain the velocity matrix elements with and without correction due to the non-local pseudopotentials.
- Run *wannier90*
- In case you want to combine two Wannierizations into a single one, use *combineW90Checkpoints.py*.
- Call *calcTB.py* to compute the Berry connection for the different schemes. This script has the option to output human-readable *wannier_tb.dat* files as well.
- The operator matrix elements can be visualized using *visualizeBZ.py*.
- When the velocity matrix elements are available, you can compare the schemes via *velocityMismatch.py*, see Eq. (57) of the manuscript.
- Calculation of dielectric constant (complex, Drude model) or optical conductivity (complex, Lorentz) can be performed using *calcEpsilon.py* and *calcSigma.py* and visualized using *visualizeEpsilon.py* and *visualizeSigma.py*.

## License
This work is published under the MIT License.
Copyright (C) 2026 IPC University of Jena

