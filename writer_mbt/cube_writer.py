#!/usr/bin/env python3
import numpy as np
from scipy.special import sph_harm, factorial, factorial2, lpmv
from tools_mbt.misc import *
from tools_mbt.grids import *
from copy import deepcopy

class Write_cube(object):

    def __init__(self, 
                 mol,
                 grid = [40, 40, 40],
                 data = "orbitals",
                 n_mo = 0,
                 output = "output.cube"):
        """
        Class to write down cube files. Keep the same standard as 
        the other writers
        """
        # Generate a Cube object
        try:
            self.mol = deepcopy(mol)
        except:
            print('Warning: Mol object cannot be deep copied. \
                  \nA pointer will be used instead. However,\
                  \nchanges will occur in place!')
            self.mol = mol

        self.cube = Cube_mol(self.mol, grid, data, n_mo)
        self.cube.generate_data(self.mol)
        self.output = output

    def write_output(self):
        """Write cube file"""

        file_name = self.output
        natoms    = self.mol.natoms
        #center    = self.cube.center
        grid      = self.cube.grid
        voxel     = self.cube.V # Voxel widths

        # Center cell at the center of geometry of the molecule
        center    = np.array([-voxel[i]*grid[i]/2 for i in range(len(grid))]) + self.cube.center 
        charges   = [0.0 for i in range(natoms)]
        if(hasattr(self.mol, "atomcharges")):
            charges = self.mol.atomcharges


        with open(file_name, "w") as f:
            f.write("Test line 1\n")
            f.write("Test line 2\n")
            # Write cell parameters and geometry
            fmt = "{:>5d}" + 3*"{:>12.6f}" + "\n"
            f.write(fmt.format(natoms, *center))
            f.write(fmt.format(grid[0], voxel[0], 0.0, 0.0))
            f.write(fmt.format(grid[1], 0.0, voxel[1], 0.0))
            f.write(fmt.format(grid[2], 0.0, 0.0, voxel[2]))
            # Write geometry
            fmt = "{:>5d}" + 4*"{:>12.6f}" + "\n"
            for i, ixyz in enumerate(self.mol.atomcoords):
                atnum = self.mol.atomnos[i]
                f.write(fmt.format(atnum, charges[i], *ixyz))
            # Write grid
            for i in range(len(self.cube.X)):
                for j in range(len(self.cube.Y)):
                    for k in range(len(self.cube.Z)):
                        f.write("{:>14.6E}".format(self.cube.volume[i][j][k]))
                        if(k%6 == 5):
                            f.write("\n")
                    f.write("\n")

if(__name__=="__main__"):
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

