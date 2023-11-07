#!/usr/bin/env python3
import numpy as np
from math import pi, factorial
from scipy.linalg import block_diag
from tools_mbt.misc import *
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.geom import Align, rotate_matrix
from tools_mbt.orbitals import rot_to_euler, \
        wigner_D
from tools_mbt.permutations import permute_orbitals,\
        h5_to_pyscf
from collections import OrderedDict
from copy import deepcopy
import h5py

class HDF5_parser(object):

    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = "spherical"):
        """HDF5 (h5) parser class. This is different
           w.r.to other classes as h5 data is already 
           brought in in memory. Transform h5 data from
           the data object to a Mol object.
        """
        # Headers, to evidence which
        # data can be retrieved
        self.__headers = OrderedDict({"CENTER_COORDINATES": False, # It implies that CENTER_ATOMNUMS IS ALSO PRESENT
            "BASIS_FUNCTION_IDS": False, # Basis+ primitives
            "MO_ENERGIES":  False,
            "MO_OCCUPATIONS":  False,
            "MO_VECTORS":  False})
        # HDF5 Parser, calle "lines" for consistency
        self.lines = h5py.File(infile, "r")

        # Update self.headers, according to the
        # headers in self.lines
        self.update_headers()

        self.__parsers = {"CENTER_COORDINATES": self.parse_coords,
                          "BASIS_FUNCTION_IDS": self.parse_shells,
#                          "MO_ENERGIES": self.parse_alpha_energies,
                          "MO_ENERGIES": self.get_mo_prop,
                          "MO_VECTORS": self.parse_alpha_mos}

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
            if key in self.lines.keys():
                # Grab first appearance
                self.__headers[key] = True

    def parse_coords(self):
        """Parse general info about the molecule"""
        self.atomcoords = np.array(self.lines["CENTER_COORDINATES"])
#        self.atomnos    = list(self.lines["CENTER_LABELS"]) 
        self.atomnos    = [atom_nums[iat.decode('ascii')[0]] for iat in list(self.lines["CENTER_LABELS"])]
        self.natoms     = len(self.atomnos)

        # Initialize _atm and _env
        for atid, ixyz in enumerate(self.atomcoords):
            atnum = self.atomnos[atid]

            self._atm.append([atnum, 20 + 4*atid, 1,\
                    20 + 4*atid + 3, 0, 0]) # add "charge"
            self._env = self._env + list(ixyz) + [0.0]
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
        atoms, env saves the primitives only once
        and bas and atm point to these same positions
        """
        prm       = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]

        # IDs of nonzero primitive coeffs.
        prim_ids = np.where(self.lines["PRIMITIVES"][:, 1] != 0)[0]
        # Retrieve number of primitives and of shells
        nprm = len(prim_ids)
        nshls = 0
        # Indices of QM atoms
        qm_atom_ids = np.unique(self.lines['BASIS_FUNCTION_IDS'][:, 0])
        # Shell to atom-index pointer
        shl_atm = []
        # Number of primitives per shell
        prm_per_shl = []
        # Angular momentum per shell
        amoms = []

#        for i in range(self.natoms):
        for i, iat in enumerate(qm_atom_ids):
            # BUGFIX: iat need not be the index that appears in the
            # definitions of at_shl and at_prm -> the indexing could start
            # from a number other than 1! The problem is that here there's no
            # distinction between QM and MM atoms
#            iat = i + 1
            # BASIS_FUNCTION_IDS for ith+1 atom
            at_shl = self.lines["BASIS_FUNCTION_IDS"][\
                    np.where(self.lines[\
                    "BASIS_FUNCTION_IDS"][:, 0] == iat)[0]]
            # PRIMITIVE_IDS for ith+1 atom, excluding
            # primitives with coeff == 0
            at_prm = self.lines["PRIMITIVE_IDS"][prim_ids][self.lines["PRIMITIVE_IDS"][prim_ids][:, 0] == iat]
            max_amom = np.max(at_shl[:, 2])
            for j in range(max_amom + 1):
                ishls =  np.max(at_shl[np.where(\
                         at_shl[:, 2] == j)][:, 1])
                nshls += ishls
#                nshls += np.max(at_shl[np.where(\
#                        at_shl[:, 2] == j)][:, 1])
                shl_atm += ishls*[i] # shell to atom map
                amoms   += ishls*[j] # Ang mom per shell

                # Get primitives per shell. Filter by 
                # amom (col 1) and by index of basis in
                # ang mom shell
                prm_per_shl += [len(at_prm[\
                        (at_prm[:, 1] == j) \
                        & (at_prm[:, 2] == k+1)])\
                        for k in range(ishls)]

        # Angular momentum per primitive
        amom_pr = []
        for i in range(len(amoms)):
            amom_pr += prm_per_shl[i]*[amoms[i]]
#        print("len(amoms)", len(amoms))
#        print("len(prm_per_shl)", len(prm_per_shl))
#        print("len(amom_pr)", len(amom_pr))

        expn     = self.lines["PRIMITIVES"][prim_ids][:,0].tolist() 
        coef     = self.lines["PRIMITIVES"][prim_ids][:,1].tolist()
#        print("len(expn) = ", len(expn))

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
           supported. The input ordering in the hdf5 file is permutted
           to the ordering indicated in the self.order 
           attribute (default = pyscf). 
        """
        nmos = int(np.sqrt(len(self.lines["MO_VECTORS"])))
        self.nmo = nmos

        # Permute AOs so as to follow the self.order
        # ordering
        self.C_mo = np.array(self.lines["MO_VECTORS"]).\
                reshape((nmos, nmos))
        # Reindex from h5 to pyscf
        perm_id = h5_to_pyscf(self.lines["BASIS_FUNCTION_IDS"])
#        perm_id = permute_orbitals(self._bas,\
#                order = "h5", target = self.order)
        self.C_mo = self.C_mo[:, perm_id]

    def parse_alpha_energies(self):
        """Parse Alpha MO energies"""
        self.moenergies = np.array(self.lines["MO_ENERGIES"])

    def parse_mo_occupations(self):
        """"""
        self.mooccnos = np.array(self.lines["MO_OCCUPATIONS"])

    def parse_alpha_spin(self):
        """"""
        try:
            self.mospin = np.array(self.lines["MO_SPIN"])
        except:
            print("MO spin set to Alpha")
            self.mospin = np.array(["Alpha" for i in range(len(self.moenergies))])

    def get_mo_prop(self):
        """Get MO properties such as Energies,
           spin and occupation numbers
        """
        self.parse_alpha_energies()
        self.parse_mo_occupations()
        self.parse_alpha_spin()

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

if(__name__ == "__main__"):
    import sys

    infile = sys.argv[1]
    mol = HDF5_parser(infile)
    mol.parse()

