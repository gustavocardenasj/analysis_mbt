#!/usr/bin/env python3
import numpy as np
import re
from math import pi, factorial
from scipy.linalg import block_diag
from tools_mbt.misc import *
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.geom import Align, rotate_matrix
from tools_mbt.orbitals import rot_to_euler, \
        wigner_D
from tools_mbt.permutations import permute_orbitals
from collections import OrderedDict
from copy import deepcopy
from parser_mbt.molden_parser import Molden_parser


class Orca_parser(object):


    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = "spherical", wf = ""):
        """
        Class constructor for the orca parser.
        wf = Wavefunction (molden) file, containing Post-SCF data. 
             Default = '' (i.e. read wf from output)
        """

        self.__headers = OrderedDict({"CARTESIAN COORDINATES (A.U.)": False,
                                      "BASIS SET IN INPUT FORMAT": False,
                                      "MOLECULAR ORBITALS": False,
                                      "FINAL SINGLE POINT ENERGY": False,
                                      "General Settings:": False
                                      })
        # Lines from file
        with open(infile, "r") as f:
            self.lines = f.read().splitlines()

        # Orca_parsr inherits from the (orca) molden parser.
        # If a complementary wf file is provided, the wavefunction
        # is read from that file
        self.wf = wf
        print("\n-----Reading molecular properties from an ORCA calculation-----")
        if(self.wf != ""):
            print("     Molecular scalar properties from: {}".format(infile))
            print("     Molecular orbitals and electron density from: {}\n".format(wf))
#            self.orca_molden = True
        else:
            print("     WARNING: Reading all molecular properties from an ORCA output file: {}".format(infile))
            print("     Only SCF (HF or DFT) molecular orbitals and electron densities available\n")
#            self.lines = self.lines
#            self.update_headers()
        self.update_headers()

        self.__parsers = {"CARTESIAN COORDINATES (A.U.)": self.parse_coords,
                          "BASIS SET IN INPUT FORMAT": self.parse_shells,
                          "MOLECULAR ORBITALS": self.get_mos,
                          "FINAL SINGLE POINT ENERGY": self.get_energies,
                          "General Settings:": self.parse_general}

        # Complementary parsers, in case a molden file is provided
        self.__cpl_parsers = {"CARTESIAN COORDINATES (A.U.)": self.parse_coords,
                              "BASIS SET IN INPUT FORMAT": self.parse_shells,
                              "FINAL SINGLE POINT ENERGY": self.get_energies,
                              "General Settings:": self.parse_general}

        # basic attributes
        self.atomcoords  = []
#        self.atomweights = []
        self.ghost       = [] # Ghost atoms
        self.freq        = []
        self.nmodes      = []
        self.atomnos     = []
        self.order       = order # Ordering of AO. Default = pyscf 
        self.basis       = basis
        self.ncrt        = 0 # Number of cartesian basis functions
        self.nsph        = 0 # Number of real spherical basis functions
        self.nmo         = 0 # Number of MOs
        self.charge      = charge
        self.mult        = mult
        self.natoms      = 0
        self.haenergies  = [] # Abs. Energy in Ha
        self.energies    = [] # Relative Energy in eV
        self.iroot       = 1 # Electronic state of interest
#        # Assess whether this is an orca molden file
#        # (in which primitives are normalized)
#        if("Molden file created by orca_2mkl" in self.lines[2]):
#            self.orca_molden = True
#        else:
#            self.orca_molden = False


        # Input arrays for integrals
        self._bas = []
        self._atm = []
        self._env = [0.0 for i in range(0,20)]

        # Other
        self._pseudonorm = np.array([]) # Pseudonormalization factors

    def parse(self):
        """General parser"""
#        if(self.wf != ""):
#            # Parsing wf data from molden file
#            self._molden = Molden_parser(self.wf, self.charge, self.mult, self.order, self.basis)
#            self._molden.parse()
#            # Copy attributes to self
#            for iattr in self._molden.__dict__.keys():
#                self[iattr] = deepcopy(self._molden.__dict__[iattr])
#            # Delete molden object
#            del self._molden
#
#            self.update_headers()
#
#            # Finally, parse complementary data
#            for key, val in self.__headers.items():
#                if(val and key in self.__cpl_parsers.keys()):
#                    # Call corresponding parser
#                    self.__parsers[key]()
#
#        else:
#            for key, val in self.__headers.items():
#                if(val and key in self.__parsers.keys()):
#                    # Call corresponding parser
#                    self.__parsers[key]()
        # Parse all info from output file
#        for key, val in self.__headers.items():
#            if(val and key in self.__parsers.keys()):
#                # Call corresponding parser
#                self.__parsers[key]()
        # Then update wavefunction info from molden file
        if(self.wf != ""):
            # Parsing wf data from molden file.
            # Also, define parser list accordingly
            parser_list = self.__cpl_parsers.keys()
            molden_obj = Molden_parser(self.wf, self.charge, self.mult, self.order, self.basis)
            molden_obj.parse()
            self.update_wf(molden_obj)
        else:
            parser_list = self.__parsers.keys()
        # Parse all info from output file
        for key, val in self.__headers.items():
            if(val and key in parser_list):
                # Call corresponding parser
                self.__parsers[key]()

    def update_wf(self, molden_obj):
        """Update wavefunction info from molden file on self.wf"""
        wf_attributes = ["C_mo","moenergies","mooccnos","mospin", "D_ao"]
        for iattr in wf_attributes:
            if(hasattr(molden_obj, iattr)):
                self[iattr] = deepcopy(molden_obj[iattr])
    
    def update_headers(self):
        """"""
        keys = self.__headers.keys()
        for key in keys:
#            blist = [key in iline.upper() for iline in self.lines]
            blist = [key in iline for iline in self.lines]
            if(True in blist):
                # Grab first appearance
                self.__headers[key] = True
    
    def parse_general(self):
        """Parse general molecular properties"""
        prop = {"Charge": "charge",
                "Mult": "mult",
                "NEL": "nele",
                "Dim": "nao"}
        for i, iline in enumerate(self.lines):
            if("General Settings:" in iline):
                for j, jline in enumerate(self.lines[i+1:]):
                    if(len(jline) == 0): break
                    row = jline.split()
#                    if(row[0] in prop.keys()):
                    key = row[-3]
                    if(key in prop.keys()):
                        self[prop[key]] = int(row[-1])
                break

    def parse_coords(self):
        """
        Parse general info about the molecule.
        Read directly A.U. entry, which also contains information
        regarding atomic numbers, weights, fragments and ghost atoms."""
        for cnt1, iline in enumerate(self.lines):
#            if("CARTESIAN COORDINATES (ANGSTROEM)" in iline):
#                for atid, jline in enumerate(self.lines[cnt1 + 2:]):
#                    if(len(jline) == 0): break
#
#                    ixyz  = [ag2bohr * float(k) for k in jline.split()[-3:]]
#                    atnum = atom_nums[jline.split()[0]]
#                    self.atomcoords.append(ixyz)
#                    self._atm.append([atnum, 20 + 4*atid, 1,\
#                                      20 + 4*atid + 3, 0, 0]) # add "charge"
#                    self._env = self._env + ixyz + [0.0] # add "charge"
#                break
            if("CARTESIAN COORDINATES (A.U.)" in iline):
                for atid, jline in enumerate(self.lines[cnt1 + 3:]):
                    if(len(jline) == 0): break
                    row = jline.split()

                    ixyz  = [float(k) for k in row[-3:]]
                    if(int(float(row[2])) == 0):
                        self.ghost.append(atid) #Keep track of ghost atoms
                    atnum = atom_nums[row[1]]
#                    atnum = int(float(row[2]))
#                    atmw  = float(row[4])
                    self.atomcoords.append(ixyz)
#                    self.atomweights.append(atmw)
                    self._atm.append([atnum, 20 + 4*atid, 1,\
                                      20 + 4*atid + 3, 0, 0]) # add "charge"
                    self._env = self._env + ixyz + [0.0] # add "charge"
                break
        self.atomcoords  = np.array(self.atomcoords).astype("float64")
#        self.atomweights = np.array(self.atomweights).astype("float64")
        # Get atomic numbers
        self._atm    = np.array(self._atm)
        self.atomnos = [iat for iat in self._atm[:,0]]
        self.natoms  = len(self.atomnos)

    def parse_shells(self):
        """Parse basis functions"""
        prm       = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]
        iatom     = 0  # Atom index
        iprim     = 0  # Primitive index
        skiplines = 0  # Skip primitive lines after reading them 
        elem_bas  = {} # Keep track of shells per element
        amom_bas  = {} # Keep track of angular momentum per shell

        for i, iline in enumerate(self.lines):
            # Pre-processing
            if("BASIS SET IN INPUT FORMAT" in iline):
                # Pre-process element-wise basis sets
                for j, jline in enumerate(self.lines[i+3:]):
                    # Parse basis set of an element
                    if("Basis set for element" in jline):
                        elem = jline.split()[-1]
                        elem_num = atom_nums[elem]
                        elem_bas[elem_num] = []
                        amom_bas[elem_num] = []
                        for k, kline in enumerate(self.lines[i+3+j+2:]):
                            if("end;" in kline): break
                            # Iterate over contractions
                            row = kline.split()
                            if(row[0].lower() in ang_mom.keys()):
                                amom   = ang_mom[row[0].lower()]
                                nprm   = int(row[1])
                                id_cnt = i+3+j+2+k+1
                                iprm   = []
                                icf    = []
                                # Iterate over primitives
                                for l, lline in enumerate(self.lines[id_cnt:id_cnt + nprm]):
                                    p1, c1 = [float(m) for m in lline.split()[-2:]]
                                    c1 = gto_sph_norm(amom,p1) * c1
                                    iprm.append(p1)
                                    icf.append(c1)
                                elem_bas[elem_num].append(iprm + icf)
                                amom_bas[elem_num].append(amom)
                    if("--------------------" in jline):break
                break

        # Now define atom-wise basis set
        for i, iatom in enumerate(self.atomnos):
            for j, jbas in enumerate(elem_bas[iatom]):
                Nprm  = int(len(jbas)/2) # Number of primitives in current shell
                self._env += jbas
                idexp  = len(self._env) - 2*Nprm
                idcf   = len(self._env) - Nprm
                amom   = amom_bas[iatom][j]
                self._bas.append([i, amom, Nprm,
                                  1, 0, idexp, idcf, 0])
                self._pseudonorm = np.concatenate((\
                        self._pseudonorm, [gto_sph_norm(amom,p1) for p1 in jbas[:Nprm]]))

        # Get number of basis functions
        self.get_num_ct()
        self.get_num_sp()

        # Re-define _bas and _env as numpy arrays
        self._bas = np.array(self._bas)
        self._env = np.array(self._env)

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

    # MO- referred methods
#    def allocate_mo(self, unrestr = False, basis = "spherical"):
    def allocate_mo(self, unrestr = False):
        if(self.basis == "spherical"):
#            dim = get_num_sp(self._bas)
            dim = self.nsph
        elif(self.basis == "cartesian"):
#            dim = get_num_ct(self._bas)
            dim = self.ncrt
        else:
            raise ValueError("basis can only be either spherical or cartesian")
        if(unrestr == True):
            self.C_alpha = np.zeros((dim,dim))
            self.C_beta = np.zeros((dim,dim))
            self.C_mo = {"alpha": self.C_alpha,
                      "beta": self.C_beta}
        else:
            self.C_mo = np.zeros((dim,dim))
        self.nmo = dim
    
    def get_all_mo(self, unrestr = False):
        """Handle reading of MOs"""
        # Unrestricted to be implemented
        self._get_mo()

    def _get_mo(self, unrestr = False):
        """Read MO coefficients"""
        parser = "MOLECULAR ORBITALS"
        if(self.basis == "cartesian"):
            nao = self.ncrt
        else:
            nao = self.nsph

        self.moenergies = []
        self.mooccnos   = []
        if(not unrestr):
            self.mospin = ["Alpha" for i in range(self.nmo)]
        else:
            self.mospin = []

        for i, iline in enumerate(self.lines):
            if(parser in iline):
                for j, jline in enumerate(self.lines[i+2:]):
                    if(len(jline) == 0): break
                    row = jline.split()
                    # Row of MOs found
                    if(row[0].isdigit()):
                        # Get MO energies and MO occupations
                        col_idx = [int(k) for k in row]
                        row_ene = self.lines[i+2+j+1].split()
                        row_occ = self.lines[i+2+j+2].split()
                        self.moenergies += [float(k) for k in row_ene]
                        self.mooccnos   += [float(k) for k in row_occ]
                        curr_id = i+2+j+4
                        # Iterate over AOs
                        for k, kline in enumerate(self.lines[curr_id:curr_id + nao]):
#                            row_coef = [float(l) for l in re.findall("[- ]\d*\.?\d+", kline)[2:]]
                            pre_coef = kline.split()[2:] # Exclude index and basis function name
                            row_coef = [float(l) for l in re.findall("-?\d+\.?\d+m?", " ".join(pre_coef))]
                            self.C_mo[col_idx, k] = row_coef #MOs are rows
                break
        self.moenergies = np.array(self.moenergies)
        self.mooccnos   = np.array(self.mooccnos)  
        self.mospin     = np.array(self.mospin)   
        # Permute AOs so as to follow the self.order
        # ordering
        perm_id = permute_orbitals(self._bas,\
                order = "orca", target = self.order, basis = self.basis)
        self.C_mo = self.C_mo[:,perm_id]
        
        # Update f and g orbitals in the case of spherical functions
        if(self.basis == "spherical"):
            print("\n-----Reading ORCA molden file-----")
            print("     Updating MOs to adapt to Orca format\n")
            self.get_orca_mo()
    
    def get_orca_mo(self):
        """
        Adapt MO coefficients to Orca convention in a spherical basis:

        Coefficients with inverted sign:
        F(+3), F(-3), G(+3), G(-3) G(+4), G(-4)
        """

        cnt_basis = 0
        # Iterate over shells
        for i, ibas in enumerate(self._bas):
            # Iterate over basis functions. Assume PySCF ordering
            amom       = ibas[1]
            amom_shl = ang_mom_sp[amom]
            for j in range(amom_shl):
                if((amom in [3,4]) and (j == 0 or j == amom_shl - 1)):
                    self.C_mo[:,cnt_basis] = -1 * self.C_mo[:,cnt_basis]
                cnt_basis += 1

    def get_energies(self):
        """Parse SCF, post-SCF and excited state energies"""

        for i, iline in enumerate(self.lines):
            # Parse energies in hartrees, except for self.energies
            if("TOTAL SCF ENERGY" in iline):
                etype = "scf_energy"
                self[etype] = float(self.lines[i+3].split()[3])
            if("MP2 TOTAL ENERGY" in iline):
                etype = "mp2_energy"
                self[etype] = float(self.lines[i].split()[3])
            if("COUPLED CLUSTER ENERGY" in iline):
                etype = "cc_energy"
                self[etype] = float(self.lines[i+7].split()[2])

            # Final single point E
            if("FINAL SINGLE POINT ENERGY" in iline):
                ha_ene = float(iline.split()[4])
                self.haenergies.append(ha_ene)
                self.energies.append(0) # Relative energy in eV

    def get_mos(self, unrestr = False):
        """Allocate MO arrays and retrieve corresponding values"""
        # Get MOs
        self.allocate_mo(unrestr)
        self.get_all_mo(unrestr)

    def __getattribute__(self, name):
        return(object.__getattribute__(self, name))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return(object.__getattribute__(self, name))











                


