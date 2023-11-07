#!/usr/bin/env python3
import numpy as np
from scipy.special import sph_harm, factorial, factorial2, lpmv
from tools_mbt.misc import *
from tools_mbt.grids import *
from copy import deepcopy
from integrals_mbt.ovlp_wrapper import get_cartnorm # Cartesian normalization matrix 
from writer_mbt.cube_writer import Write_cube


from argparse import ArgumentParser as ap
from molecule import Mol

# Module to generate cube files for arbitrary AO/MO

parser = ap("Generate a cube file for the electron density OR for a specific orbital in a molecular data file.")
parser.set_defaults(norb = 0, basis = "spherical", grid = [40, 40, 40], data = "orbitals", output = "output.cube")
parser.add_argument("-f", dest = "infile", type = str,\
        help = "Provide file containing Orbital Info")
parser.add_argument("-b", dest = "basis", type = str,\
        help = "Type of basis set. Default = spherical")
parser.add_argument("-g", dest = "grid", nargs = "+",\
        help = "Grid size; provide either 1 or 3 arguments. Default = 40 40 40")
parser.add_argument("-d", dest = "data", type = str,\
        help = "Type of data to be printed. Default = orbitals")
parser.add_argument("-n", dest = "norb", type = int,\
        help = "Number/index of the MO to be printed. Default = 0")
parser.add_argument("-o", dest = "output", type = str,\
        help = "Output name. Default = output.cube")


options = parser.parse_args()
infile  = options.infile
data    = options.data
n_mo    = options.norb
basis   = options.basis
output  = options.output
if(len(options.grid) > 1):
    grid = [int(ig) for ig in options.grid]
elif(len(options.grid) == 1):
    grid = [int(options.grid) for i in range(3)]
else:
    raise ValueError("Select a right number of elements for grid")

# Load molecule data
mol = Mol(infile, basis = basis)
mol.parse()
writer = Write_cube(mol, grid, data, n_mo, output)
writer.write_output()

