# Analysis MBT
This is a set of analysis tools for a selected number of molecular files containing structural and wavefunction information. This is an alpha version of a module that will be incorporated in MoBioTools, please use with caution. These tools can be used as stand-alone scripts (see scripts folder) or an an API via the Mol() class.

# Capabilities

Interconversion among wavefunction formats compatible with different quantum chemistry packages:
- Gaussian fchk (r/w)
- Molden files (r/w); Orca molden (r)
- Orca output file (w) (To be used in conjunction with Orca molden for post-SCF properties)
- Molcas h5 (r/w)
- Molcas InpOrb (w)
- Gamess output files (r)
- Games .dat files (r/w)

Analysis of molecular coordinates, atomic and molecular orbitals, molecular energies, transition energies, point charges, electrostatic potential...

Rotation (and alignment) of molecular structures AND molecular orbitals

Writing (Gaussian) Cube files containing MO or ESP volumetric data

...

# Requirements

Python >= 3.6

NumPy

SciPy

# Installation

Download the repository files, then update the following environment variables:

```
export ANALYSIS_MBT=/path/to/repo/files
export PYTHONPATH=$ANALYSIS_MBT:$PYTHONPATH
export PATH=$ANALYSIS_MBT/scripts_mbt:$PATH
```

# Usage: As a toolkit
Via the scripts present in the /scripts folder.

- Change format

```
change_format.py -f butadiene1.inp.scf.molden -o fchk -b cartesian -n butadiene1.inp.scf
```

where
```
-f = input file
-o = format of the output file
-b = basis type (either cartesian or spherical)
-n = name of the output file without extension
```


Multiple input files: Orca outputs. 

It is possible to print the basis functions and the MOs (and the electron densities) 
during an Orca calculation in the corresponding output file by adding the options 

```
! PrintMOs PrintBasis
```

and 

```
! Largeprint
```

to include the above plus the electron densities and more verbose output. HOWEVER it only prints
MOs stemming from an SCF calculation (HF or DFT). For post-SCF computations, the natural orbitals
and the electron densities are printed on separate binary files upon explicit request from the user
in the corresponding section (e.g., %mp2...end or %mdci...end). These binaries can be transformed in
orca molden files using the orca_2mkl tool. See the Orca documentation for further details.

To analyze post-SCF molecular properties with the present toolkit, one can either provide the wavefunction (wf)
file only as input:

```
change_format.py -f dimer_h2.mp2nat.molden.input -o fchk -b spherical -n dimer_h2 -d MP2
```

or both the wf and the output file. The latter bears information not present in the molden file (e.g., energies,
ghost atoms, etc...)

```
change_format.py -f dimer_h2.inp.out  -o fchk -b spherical -n dimer_h2 -d MP2 -wf dimer_h2.mp2nat.molden.input
```

- Write cube file

Electron density

```
write_cube.py -f butadiene1.inp.scf.molden -b cartesian -d density 
```

where
```
-f = input file
-b = basis type (either cartesian or spherical)
-d = type of data (either density or orbitals)
```

Molecular orbitals
write_cube.py -f butadiene1.inp.scf.molden -b cartesian -d orbitals -n 14

where (repeated args. are the same as above)
-n = index (number) of the MO of interest

- Compute AO and MO overlap matrices

```
overlaps.py -f1 butanol1.inp.scf.molden -f2 butanol2.inp.scf.molden -b spherical -r 0
```

where
```
-f1 = input file 1
-f2 = input file 2
-b = basis type (either cartesian or spherical)
-r = rotate molecule f2 in-place to align it with f1. Either 0 (False) or 1 (True). If 0, create a deepcopy of a Mol() object of f2.
```

This script will generate two serialized pickle files: Sao.pickle and Smo.pickle, containing the AO and MO overlap matrices, respectively.

All of the above scripts can also be executed in an interactive shell (e.g., the IPython kernel), so that the defined data can be further manipulated. See also API below.

# Usage: As an API
The main class is the Mol() class, defined in molecule.py. If all of the environment variables have been correctly defined, it can be imported as
```
from molecule import Mol
```

Retrieve molecular data - provide input file name and type of basis ("cartesian" or "spherical"):
```
mol = Mol("butadiene1.inp.scf.molden", "cartesian")
```

All of the attributes of the mol object can be visualized in the usual manner:
```
print(mol.__dict__.keys())
```

# Notes:
The molden files shown above can be found in the /samples folder; they have been sorted out depending on the type of basis set used for the calculation (cartesian or spherical)

# Caveats:
- For the time being, it is only compatible with closed shell systems.
- MM point charges are not read (only QM data)
- The coordinates are defined as a (N,3) array, where N = number of atoms
- The MOs are accessed via the C_mo attribute (e.g., mol.C_mo) IMPORTANT: The ordering is the transpose with respect to that appearing on several textbooks (e.g., Szabo), so that:

-- Row indices = MOs
-- Column indices = AOs

So that, for example, the MO overlap matrix between, say two geometries defined by the mol1 and mol2 objects,should be computed as follows (pseudocode):
 
```
Smo = mol1.C_mo * Sao * transpose(mol2.C_mo)
```

- The shared object intwrap.so may not work in all Linux distributions and would thus need to be recompiled. The compilation is a 2 step process:
1)
```
g++ -fPIC -c intwrap.cpp
```

2)
```
g++ -shared intwrap.o -o intwrap.so
```



- pyoverlaps.py NEEDS to be updated and is thus, not usable! Pleae refer to the MoBioTools project instead.
