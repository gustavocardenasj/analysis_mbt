#!/usr/bin/env python3
import numpy as np
from math import pi, factorial
from scipy.linalg import block_diag
from tools_mbt.misc import *
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.misc import norm_cart as cnorm
from tools_mbt.geom import Align, rotate_matrix
from tools_mbt.orbitals import rot_to_euler, \
        wigner_D, euler_to_rot, cartesian_D
from tools_mbt.permutations import permute_orbitals, \
        permute_matrix, perm_mixed_matrix

# Parsers
from parser_mbt.molden_parser import Molden_parser
from parser_mbt.fchk_parser import Fchk_parser
from parser_mbt.hdf5_parser import HDF5_parser
from parser_mbt.gamess_parser import Gamess_parser
from parser_mbt.orca_parser import Orca_parser
from parser_mbt.psi4_parser import Psi4_parser

# Writers
from writer_mbt.gamess_writer import *
from writer_mbt.inporb_writer import *
from writer_mbt.molden_writer import *
from writer_mbt.fchk_writer import *

try:
    import h5py
except:
    print("Warning: HDF5 format is not supported")


FORMAT_PARSER = {"Molden Format": "molden",
                 "Number of atoms                            I": "fchk",
                 "CENTER_COORDINATES": "h5",
                 "GAMESS execution script": "gamess",
                 "Directorship and core code : Frank Neese": "orca",
                 "Psi4: An Open-Source Ab Initio Electronic Structure Package": "psi4"}

class Mol(object):

    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = "spherical", center_origin = False, density = "None", wf = ""):
        """Class to defined a Mol object, bearing general molecular
           data. Arguments:
           infile = input file, str. Mandatory
           charge 
           mult   = spin multiplicity
           order  = ordering of atomic orbitals. Default = pyscf. Other 
                    options: molden, molcas, inporb
           basis  = cartesian or spherical, Default = spherical
           center_origin = Center molecule at the origin. Default = False
           density = Whether to compute the density matrix from the MOs.
                     e.g.: SCF, MP2, CCSD. Default = None

        """
        try:
            with open(infile, "r") as f:
                self.lines = f.read().splitlines()
            print("\n-----Reading ASCII input file-----\n")
        except:
            self.lines = h5py.File(infile, "r")
            print("\n-----Reading HDF5 input file-----\n")

        self.format = self.infer_format()
        self.center_origin = center_origin
        self.orca_molden = False # Assume false by default
        self.density = density
        self._density_parser = {"SCF": "Total SCF Density",
                                "MP2": "Total MP2 Density",
                                "CCSD": "Total CC Density"}

        if(self.format == "molden"):
            self._parser = Molden_parser(infile, charge, mult, order, basis)
        elif(self.format == "fchk"):
            self._parser = Fchk_parser(infile, charge, mult, order, basis)
        elif(self.format == "h5"):
            self._parser = HDF5_parser(infile, charge, mult, order, basis)
        elif(self.format == "gamess"):
            self._parser = Gamess_parser(infile, charge, mult, order, basis)
        elif(self.format == "orca"):
            self._parser = Orca_parser(infile, charge, mult, order, basis, wf)
        elif(self.format == "psi4"):
            self._parser = Psi4_parser(infile, charge, mult, order, basis, wf)
        else:
            self._parser = None

    def parse(self):
        """Parse various attributes from input file"""
        self._parser.parse()
        # Create a deepcopy of each attribute, then destroy _parser object
        for iattr in self._parser.__dict__.keys():
            self[iattr] = self._parser.__dict__[iattr]

        # Assess whether to center at the origin
        if(self.center_origin):
            centered = self.translate_mol([0., 0., 0.])

        # Compute density matrix upon request
        if(self.density != "None"):
            print("\n-----Density Matrix computation requested-----")
            print("     I will attempt to compute the {} density matrix from the MOs...".format(self.density))
            try:
                self.get_density_matrix()
                print("     ...Density matrix stored in .D_ao attribute\n")
            except:
                print("     The density matrix could not be computed.\n")

    def write(self, fmt = "molden", outfile = "outfile"):
        """
        Write a file with the molecular information present in the Mol() object. 

        Args:
        fmt = Format of the output file (Default = molden; also fchk, inporb, gamess_dat)
        out_name = Name of the output file. Default = outfile

        Warning: in its current implementation it will create a deepcopy of the object,
        that will only last within the scope
        """

        if(fmt == "molden"):
            writer = Write_molden(self, outfile + ".molden")
            ext    = ".molden"
        elif(fmt == "inporb"):
            writer = Write_inporb(self, outfile + ".InpOrb")
            ext    = ".InpOrb"
        elif(fmt == "gamess_dat"):
            writer = Write_gamess(self, outfile + ".dat")
            ext    = ".dat"
        elif(fmt == "fchk"):
            writer = Write_fchk(self, outfile + ".fchk")
            ext    = ".fchk"
        else:
            raise(ValueError("Format {} currently not supported.".format(fmt)))
            return

        writer.write_output()
            

    def update_coords(self, new_coords):
        """
        Update coordinates, with the array provided as input
        """
        if(hasattr(self, "atomcoords")):
            self.atomcoords = new_coords
            if(hasattr(self, "_env")):
                natom = len(self.atomcoords)
                temp = np.hstack((self.atomcoords,np.zeros((natom,1))))
                self._env[20:20 + 3*natom+natom] = temp.flatten()
            return(True)
        else:
            raise AttributeError("{} does not have attribute '{}'. Molecule will not be translated".format(self, "atomcoords"))
            return(False)

    def translate_coords(self, center):
        """
        Translate molecular coordinates to center
        """
        self.CM   = np.average(self.atomcoords, axis = 0)
        displ_vec = center - self.CM
        new_crd   = self.atomcoords + displ_vec

        coords_displaced = self.update_coords(new_crd)
        print("Coordinates displaced: {}".format(coords_displaced))
        return

    def hess_to_freq(self):
        """Compute frequencies and normal modes of vibration from the
           hessian matrix in self.hessian. Hessian in Hartree/bohr^2.
           The hessian matrix is first transformed into millidines/Angstrom,
           then mass weighted and finally diagonalized. The eigenvalues are
           then transformed into cm-1
        """
        c_light  = 2.99792458e10 # cm/s

        # Atomic weights vector
        wg = np.array([weights[mol.atomnos[int(i / 3)]] for i in range(24)])
        wg_div = np.sqrt(np.outer(wg, wg))

        if(hasattr(self, "hessian")):
            # Hartree/A^2 -> kcal/(mol A^2) -> millidines/A mass_weighted
            hess = self.hessian * ha2mdine_b /wg_div
            v, w = np.linalg.eigh(hess) # eigenvalues, eigenvectors
            self.freq = (2 * np.pi * c_light) ** -1 * np.sqrt(6.022 * 10 ** 28 * v[6:])
        else:
            raise AttributeError("Hessian matrix not present")


    # MO and rotation-related functions

    def rotate_molecule(self, R = None, alpha = None, beta = None, gamma = None):
        """Rotate molecular coordinates, AO and MO (if present)
           depending on the R 3x3 rotation matrix provided as input.
           
           Args:
           R = 3x3 rotation matrix
           alpha, beta, gamma = euler angles in the Z-Y-Z convention

           Rotations performed in-place
        """
        if(type(R) == np.ndarray):
            # Use rotation matrix provided as input.
            # Get the rotated coordinates
            pass
        elif((alpha, beta, gamma) != (None, None, None)):
            R = euler_to_rot(alpha, beta, gamma) # Update: from_right = False
        else:
            raise(ValueError("A valid rotation matrix could not be built"))

        # Rotate coordinates
        self.rotate_coords(R) # from_right = False
        # Rotate AO and MO
        self.rotate_mo(R)

    def align_molecule(self, ref_coords = []):
        """
        Align molecular coordinates and (eventually) AO and MO
        of Mol on the basis of the coordinates provided as input
        """
        if(len(ref_coords) > 0):
            # Determine rotation matrix
            rotor   = Align(ref_coords, self.atomcoords, self.atomnos)
#            rot_mat = rotor.get_rotation_matrix()
            rot_mat = rotor.get_rotation_matrix().T # Make it act from left

            # Then rotate and align coordinates
            self.rotate_molecule(rot_mat)
            # Finally translate coordinates to the Center of Geometry
            # of the reference coordinates
            com_ref = np.average(ref_coords, axis = 0)
            self.translate_coords(com_ref)
        else:
            print("Coordinates not modified")

    def get_cartnorm_abs(self, high = True, pseudonorm = False):
        """
        Get a vector with the normalization constants
        of all basis functions.

        if high == True, normalize only L>1
        if pseudonorm == True, pseudonormalization only
        """
        bas = self._bas
        nct = self.ncrt
        C   = np.identity(nct)
        C   = np.ones(nct)
        i_row = 0
        j_col = 0
        k_fnc = 0
        # Iterate over shells
        for cn1, ishl in enumerate(bas):
            # Number of cartesian functions per shell
            ict = ang_mom_ct[ishl[1]]
            if(high and ishl[1] in [0,1]):
                print("Not notmalizing bas ", ishl)
                # Do not normalize
                k_fnc += ict
                j_col += ict
                i_row += ict
                continue
            # Iterate over basis functions
            for cn2 in range(ict):
                alpha = self._env[ishl[5]]
                lx, ly, lz = ANGCRT[off_angcrt[ishl[1]] + cn2]
                C[k_fnc + cn2] = gto_cart_norm_abs(lx, ly, lz, alpha)
            k_fnc += ict
        return(np.diag(C))


    def get_ao_rotation(self, R):
        """
        Get the atomic orbital rotation matrix from
        a coordinate rotation matrix R
        """
        # Get rotation matries for l = 0,1,2,3
        if(self.basis == "cartesian"):
            alpha, beta, gamma = rot_to_euler(R)
            # Up to D functions for now
            rot_mos = [cartesian_D(i, alpha, beta, gamma) for i in range(0,3)]
        else:
            alpha, beta, gamma = rot_to_euler(R, True) # I have implemented it transposed, therefore the matrix ins transposed inside the function
            rot_mos = [wigner_D(i, alpha, beta, gamma)\
                       for i in range(0,4)]

        # Get rotation matrix or self. Matrix with
        # pyscf ordering, then permute depending on
        # the value of the self.order attribute
        mat_list = [rot_mos[i] for i in self._bas[:,1]]
        self.ao_rotation = block_diag(*mat_list)

        # Permute orbitals so that they have the 
        # PySCF ordering. Distinguish between mixed
        # (cartesian <-> spherical) and non-mixed 
        # situations
        self.before_ao_rotation = self.ao_rotation
        self.perm_id = permute_orbitals(self._bas, "pyscf", self.order, self.basis)
        self.ao_rotation = permute_matrix(self.ao_rotation, self.perm_id) # Update: The matrix is transposed back so that it acts from the left
        self.after_ao_rotation = self.ao_rotation 
        if(self.basis == "cartesian"):
            # Include normalization constants for L>1
            self.Cn_dir = self.get_cartnorm_abs()
            self.Cn_inv = np.linalg.inv(self.Cn_dir)
            self.ao_rotation = np.matmul(np.matmul(self.Cn_dir, self.ao_rotation), self.Cn_inv).T # Update: Taking the reanspose makes P work

    def rotate_coords(self, R = np.identity(3), alpha = None, beta = None, gamma = None, from_right = False):
        """
        Rotate molecular coordinates according to the rotation matrix
        provided as input. Notice that the coordinates need to be first
        centered at the origin prior to prform the rotation, so a translation
        is performed first.

        The sequence of operations is thus:

        coord = T^-1 * R * T

        where T= translation to the origin

        """
        if(from_right):
            R = R.T
        # Generate expanded matrix: add an extra column of ones
        crd_exp = np.hstack((self.atomcoords, np.ones((self.natoms,1))))
        # Generate translation matrix
        com    = np.average(self.atomcoords, axis = 0)
        T      = np.identity(4)
        T[:,3] = [i for i in com] + [1.]
        Tinv   = np.linalg.inv(T)

        # Expand rotation matrix
        R_exp = np.identity(4)
        R_exp[:3,:3] = R
        # Now get rotated coordinates (extended version)

        rot_exp = np.einsum("ij,jk,kl,lm->im", Tinv, R_exp, T, crd_exp.T).T
        print(rot_exp)

        # Fianlly, update atomcoors and _env
        self.update_coords(rot_exp[:,:3])
    
    def rotate_mo(self, R = None, alpha = None, beta = None, gamma = None):
        """
        Rotate molecular orbitals in correspondence 
        with a given atomic orbital rotation matrix.
        """
        if(type(R) == np.ndarray):
            pass
        elif((alpha, beta, gamma) != (None, None, None)):
            R = euler_to_rot(alpha, beta, gamma)
        else:
            raise(ValueError("A valid rotation matrix could not be built"))
        # Assess whether the AO rotation matrix has already been
        # determined
        if(not hasattr(self, "ao_rotation")):
            # Geo AO rotation matrix
            self.get_ao_rotation(R)
        
        # Now rotate MOs
        self.C_mo = np.matmul(self.ao_rotation, self.C_mo.T).T
    
    def get_density_matrix(self):
        """
        Compute the density matrix from the MO coefficients.
        For correlated methods, the MOs are assumed to be the natural orbitals
        """

        print("     Warning: Computing density matrix from the MOs provided in input.\
              \n     Hereby we assume them to be the Natural Orbitals")
        if(hasattr(self, "mooccnos")):
            print("     Occupation numbers found")
        else:
            raise(ValueError("     mooccnos (occupation numbers) not present.\
                            \n     The density matrix will not be computed\n"))
            return
        if(hasattr(self, "C_mo")):
            print("     MO coefficients found")
        else:
            raise(ValueError("     MO Coefficients not present.\
                            \n     The density matrix will not be computed"))
            return
        occup = np.diag(self.mooccnos)
        self.D_ao = np.matmul(self.C_mo.T, np.matmul(occup, self.C_mo))


    def update_order(self, target = "pyscf"):
        """Update AO ordering, according to the 
           target argument. Default: pyscf
        """
        perm_id = permute_orbitals(self._bas, self.order, target, self.basis)
        # Update ao rotation matrix, C_mo, and other
        # AO-involving matrices, if present
        if(hasattr(self, "ao_rotation")):
            self.ao_rotation = permute_matrix(self.ao_rotation, perm_id)
        if(hasattr(self, "C_mo")):
            self.C_mo = self.C_mo[:, perm_id]
        if(hasattr(self, "D_ao")):
            self.D_ao = permute_matrix(self.D_ao, perm_id)
        # Update self.order attribute
        self.order = target

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def infer_format(self):
        """Infer the format of the input file"""
        for key in FORMAT_PARSER.keys():
            bl = [key in iline for iline in self.lines]
            if(True in bl):
                return(FORMAT_PARSER[key])
        return(None)

if(__name__=="__main__"):
    import sys

    infile = sys.argv[1]
    mol = Mol(infile)
    mol.parse()

