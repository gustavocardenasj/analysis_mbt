#!/usr/bin/env python3
from molecule import Mol
from argparse import ArgumentParser as ap

parser = ap("Script to change the format of a given file containing\
             \nmolecular data")
parser.set_defaults(basis = "spherical", outfile = "outfile", density = "None", wavefunction = "")
parser.add_argument("-f", dest = "infile", type = str, help = "Input file. Supported formats: molden, molcas h5, molcas INPORB, fchk, GAMESS output file")
parser.add_argument("-o", dest = "outfmt", type = str, help = "Format of the output file. Suported formats: molden, fchk, INPORB, h5")
parser.add_argument("-b", dest = "basis", type = str, help = "Type of basis set (if applicable). Default = spherical")
parser.add_argument("-n", dest = "outfile", type = str, help = "Name without extension of the output file. Default = outfile")
parser.add_argument("-d", dest = "density", type = str, help = "Whether to print density matrix. Examples: SCF, MP2, CCSD. Default = None")
parser.add_argument("-wf", dest = "wavefunction", type = str, help = "Include extra file containing the wavefunction information (e.g., Orca molden)")

options      = parser.parse_args()
infile       = options.infile 
outfmt       = options.outfmt 
outfile      = options.outfile
basis        = options.basis
density      = options.density
wavefunction = options.wavefunction

# Retrieve data
mol = Mol(infile, basis = basis, density = density, wf = wavefunction)
mol.parse()

# Print output
mol.write(outfmt, outfile)
