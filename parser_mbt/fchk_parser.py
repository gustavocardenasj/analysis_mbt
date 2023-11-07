#!/usr/bin/env python3
import numpy as np
from math import pi, factorial
from scipy.linalg import block_diag
from tools_mbt.misc import *
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.geom import Align, rotate_matrix
from tools_mbt.orbitals import rot_to_euler, \
        wigner_D
from tools_mbt.permutations import permute_orbitals
from collections import OrderedDict


class Fchk_parser(object):

    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = "spherical"):
        """Fchk parser class"""
        # Headers, to evidence which
        # data can be retrieved

        # Each header introduces an entire "section"
        # within the fchk input file

        self.__headers = OrderedDict({"Number of atoms": False,
            "Number of contracted shells": False,
            "Alpha Orbital Energies":  False,
            "Alpha MO coefficients": False,
            "Beta Orbital Energies":  False,
            "Beta MO Coefficients": False,
            "Cartesian Gradient": False,
            "Cartesian Force Constants": False,
            })

        # Lines from infile
        with open(infile, "r") as f:
            self.lines = f.read().splitlines()

        # Update self.headers, according to the
        # headers in self.lines
        self.update_headers()

        self.__parsers = {"Number of atoms" : self.parse_coords,
                          "Cartesian Force Constants": self.parse_hessian,
                          "Number of contracted shells": self.parse_shells,
                          "Alpha Orbital Energies": self.parse_alpha_energies,
                          "Alpha MO coefficients": self.parse_alpha_mos} 

        # basic attributes
        self.atomcoords = []
        self.freq       = []
        self.nmodes     = []
        self.atomnos    = []
        self.order      = order # Ordering of AO. Default = pyscf 
        self.basis      = basis
        self.ncrt       = 0 # Number of cartesian basis functions
        self.nsph       = 0 # Number of real spherical basis functions
        self.nmo        = 0 # Number of MOs
        self.charge     = charge
        self.mult       = mult
        self.natoms     = 0

        # Input arrays for integrals
        self._bas = []
        self._atm = []
        self._env = [0.0 for i in range(0,20)]

        # Other
        self._pseudonorm = np.array([]) # Pseudonormalization factors

    def parse(self):
        """General parser"""
        for key, val in self.__headers.items():
            if(val and key in self.__parsers.keys()):
                # Call corresponding parser
                self.__parsers[key]()

    def update_headers(self):
        """"""
        keys = self.__headers.keys()
        for key in keys:
            blist = [key in iline for iline in self.lines]
            if(True in blist):
                # Grab first appearance
                self.__headers[key] = True

    def parse_coords(self):
        """Parse general info about the molecule,
           including charge, multiplicity, number of 
           electrons, coordinates, etc.
        """

        # Parse scalar values
        self.natoms = self._parse_scalar("Number of atoms")
        self.charge = self._parse_scalar("Charge ")
        self.mult = self._parse_scalar("Multiplicity")
        self.nele = self._parse_scalar("Number of electrons")
        self.aele = self._parse_scalar("Number of alpha electrons")
        self.bele = self._parse_scalar("Number of beta electrons")

        # Parse vector values
        # Coordinates. Reshape atomcoords into a 
        # (ncrd/3, 3) array
        self.atomcoords = self._parse_vector(\
                "Current cartesian coordinates").reshape((self.natoms,3))
#        self._parse_vector(\
#                "Current cartesian coordinates")
        # Atom nums
        self.atomnos = self._parse_vector("Atomic numbers")

        # Update _atm and _env
        self._get_atm_env()

    def parse_hessian(self):
        """Get cartesian force constant matrix as a rank-2 array. Units: Ha/bohr^2"""
        dim    = 3*self.natoms
        linarr = self._parse_vector("Cartesian Force Constants")
        # Transfomr into a upper triangular matrix,
        # then symmetrize
        
        ind = np.tril_indices(dim)
        outarr = np.zeros((dim, dim))
        outarr[ind] = linarr
        outarr = (outarr + outarr.T) - np.diagflat(outarr.diagonal())
        self.hessian = outarr

    def _parse_scalar(self, parser):
        """Parse an integer value from an fchk file"""
        val = 0

        for cnt, iline in enumerate(self.lines):
            if(parser in iline):
                val = int(iline.split()[-1]) 
                break
        return(val)

    def _parse_vector(self, parser):
        """Parse an array from an fchk file.
           Provide the parser line as input. Return 
           array.
        """

        outarr = []
        for cnt1, iline in enumerate(self.lines):
            if(parser in iline):
                # Length of linearized array
                nelem = int(iline.split()[-1])
                tp    = iline.split()[-3]
                if(tp == "R"):
                    outtype = "float64"
                elif(tp == "I"):
                    outtype = "int64"
                else:
                    raise TypeError("outtype not of type R or I")
                for cnt2, jline in enumerate(self.lines[cnt1 + 1:]):
                    if(len(outarr) >= nelem):
                        break
                    # Transform to float
                    curr_crd = [float(ix) for ix in jline.split()]
                    outarr += curr_crd
                break
        # Transform list into an array
        outarr = np.array(outarr).astype(outtype)
        return(outarr)

    def _get_atm_env(self):
        """Initialize _atm and include atomcoords in _env
           self.atomcoords and self.atomnos need to be defined
        """

        for atid, atnum in enumerate(self.atomnos):
            self._atm.append([atnum, 20 + 4*atid, 1,\
                            20 + 4*atid + 3, 0, 0]) # add "charge"
            ixyz = list(self.atomcoords[atid])
            self._env = self._env + ixyz + [0.0]
        self._atm = np.array(self._atm)


    def parse_shells(self):
        """
        bas: [[atom_idx, amom, n_prm, n_contr, 0, idx_frst_exp,
               idx_first_coeff, 0]..] -> ith shell
        atm: [[nucl_chg, id_x_crd, id+1 z_crd, 0, 0]]
        env: [0, ..., 0, ->0-19 idx
             x_crd_at1, y, z, 0.0, --> x, y, z , buff (chg?) atom 1
             x_crd_at2, y, z, 0.0,
             ...
             1_exp_1_shell, ..., last_exp_1_shell,
             1_cf_1_shell, ..., last_cf_1_shell,
             ...
             1_exp_last_shell, last_exp_last_shell,
             1_cf_last_shell, last_exp_last_shell]
    
        If the same primitives are used for different
        atoms, end saves the primitives only once
        and bas and atm point to these same positions
        """
        prm       = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]

        # Retrieve scalar data: N shells and N primitives
        nshls = self._parse_scalar("Number of contracted shells")
        nprm  = self._parse_scalar("Number of primitive shells")

        # Retrieve vector data:

        # Angular momentum per shell
        amoms = np.abs(self._parse_vector("Shell types"))
        # Number of primitives per shell
        prm_per_shl = self._parse_vector("Number of primitives per shell")
        
        # Angular momentum per primitive
        amom_pr = []
        for i in range(len(amoms)):
            amom_pr += prm_per_shl[i]*[amoms[i]]
        
        # Shell to atom-index pointer
        shl_atm = np.array([iat - 1 for iat in self._parse_vector("Shell to atom map")])

        expn    = self._parse_vector("Primitive exponents").tolist()
        coef    = self._parse_vector("Contraction coefficients").tolist()
        # Define pseudo-normalization array
        self._pseudonorm = np.array([gto_sph_norm(amom_pr[cnt],\
                iexp) for cnt, iexp in enumerate(expn)])

        # Psuedo-normalize coefficients
        coef = self._pseudonorm * coef
        coef = coef.tolist()

        # Iterate over shells to define _bas and _env
        offset = 0
        for i in range(nshls):
            Nprim = prm_per_shl[i]
            iexpn = expn[offset:offset + Nprim]
            icoef = coef[offset:offset + Nprim]

            iatom = shl_atm[i]
            iamom = amoms[i]
            if(iexpn + icoef not in prm):
                prm.append(iexpn + icoef)
                self._env = self._env + iexpn + icoef
                idexp      = len(self._env) - 2*Nprim
                idcf       = len(self._env) - Nprim
                self._bas.append([iatom, iamom, Nprim, 1, 0, idexp, idcf, 0])
            else:
                natom  = self.natoms
                id_prm = prm.index(iexpn + icoef) 
                flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                idexp = 20 + 4*natom + len(flat_prm)
                idcf  = idexp + Nprim
                self._bas.append([iatom, iamom, Nprim, 1, 0, idexp, idcf, 0])
           # Update offset 
            offset += Nprim

        # Get number of basis functions
        self.get_num_ct()
        self.get_num_sp()

        # Re-define _bas and _env as numpy arrays
        self._bas = np.array(self._bas)
        self._env = np.array(self._env)

    def parse_alpha_mos(self):
        """Parse Alpha MO coefficients and return a (nsph, nsph) 
           shape array. Only expansions of spherical basis functions 
           supported. The input ordering in the fchk file is permutted
           to the ordering indicated in the self.order 
           attribute (default = pyscf). 
        """
        
        nmos = int(np.sqrt(len(self._parse_vector("Alpha MO coefficients"))))
        self.C_mo = self._parse_vector("Alpha MO coefficients").\
                reshape((nmos, nmos))

        # Permute AOs so as to follow the self.order
        # ordering
        perm_id = permute_orbitals(self._bas,\
                order = "molden", target = self.order, basis = self.basis)
        self.C_mo = self.C_mo[:, perm_id]
        self.nmo  = nmos

    def parse_alpha_energies(self):
        """Parse Alpha MO energies.
        """
        self.moenergies = self._parse_vector("Alpha Orbital Energies")

    def get_num_ct(self):
        """Get the number of cartesian functions from a bas array. No return type, but instead defines the attribute ncrt. Notice that ncrt is also defined on __int__"""
        self.ncrt = 0
        for cnt, ishl in enumerate(self._bas):
            self.ncrt += ang_mom_ct[ishl[1]]
    
    def get_num_sp(self):
        """Get the number of spherical functions from a bas array. No return type, but instead defines the attribute nsph"""
        self.nsph = 0
        for cnt, ishl in enumerate(self._bas):
            self.nsph += ang_mom_sp[ishl[1]]

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

if(__name__ == "__main__"):
    import sys

    infile = sys.argv[1]
    mol = Fchk_parser(infile)
    mol.parse()

