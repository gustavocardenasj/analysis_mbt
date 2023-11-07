#!/usr/bin/env python3
from argparse import ArgumentParser as ap
import numpy as np
from molecule import Mol
from copy import deepcopy
import pickle
from integrals_mbt.ovlp_wrapper import get_ao_overlap, get_mo_overlap

parser = ap("Script to compute the (AO and MO) overlap matrix between two Mol objects")
parser.set_defaults(f2 = None, basis = "spherical", rotation = 1)
parser.add_argument("-f1", dest = "file1", type = str,\
                    help = "Input file for Mol 1")
parser.add_argument("-f2", dest = "file2", type = str,\
                    help = "Input file for Mol 2. Default = None")
parser.add_argument("-b", dest = "basis", type = str,\
                    help = "Basis type, either cartesian or spherical. Default = spherical")
parser.add_argument("-r", dest = "rotation", type = int,\
                    help = "Perform an in-place alignment of the target coordinates. Default = 1. If 0 (False) create a deepcopy of the correspondin Mol object.")
options       = parser.parse_args()
file1         = options.file1
file2         = options.file2
basis         = options.basis
rotation      = options.rotation
center_origin = False

# The targer geometry is rotated and displaced prior to performing
# the overlaps calculations. This so as to avoid eventual undesired 
# inplace rotations


mol1 = Mol(file1, basis = basis, center_origin = center_origin)
mol1.parse()
if(options.file2 != None):
    mol2 = Mol(file2, basis = basis, center_origin = center_origin)
    mol2.parse()

    if(not rotation):
        # Create deepcopy of mol2
        tgt = deepcopy(mol2)
        tgt.align_molecule(mol1.atomcoords)

        # Align tgt to mol1, leaving mol2 unchanged

        Sao = get_ao_overlap(mol1, tgt)
        Smo = get_mo_overlap(mol1, tgt)
    else:
        # Rotate molecule in-place
        mol2.align_molecule(mol1.atomcoords)
        Sao = get_ao_overlap(mol1, mol2)
        Smo = get_mo_overlap(mol1, mol2)
else:
    Sao = get_ao_overlap(mol1)
    Smo = get_mo_overlap(mol1)

# Save matrices in pickle files
with open("Sao.pickle", "wb") as f, open("Smo.pickle", "wb") as g:
    pickle.dump(Sao, f)
    pickle.dump(Smo, g)
