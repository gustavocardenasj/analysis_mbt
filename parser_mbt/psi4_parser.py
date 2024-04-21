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

class Psi4_parser(object):

    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = "spherical", wf = ""):
        """
        Class constructor for the psi4 parser.
        
        NOTE: The wavefunction information an the basis set need to be provided
        on a separate molden file, as the wf argument:
        
        wf = Wavefunction (molden) file, containing Post-SCF data. 
             Default = '' (i.e. read wf from output)
        """
        self.__headers = OrderedDict({"Geometry (in Angstrom)": False,
                                      "==> Primary Basis <==":  False,
                                      "Final Energy:":          False,
                                      "Charge       =":         False})

        # Lines from file
        with open(infile, "r") as f:
            self.lines = f.read().splitlines()
        
        self.update_headers()
        
        self.__parsers = {"Geometry (in Angstrom)": self.parse_coords,
                          "==> Primary Basis <==": self.parse_shells,
                          "Final Energy:": self.get_energies,
                          "Charge       =": self.parse_general}

        # basic attributes
        self.atomcoords  = []
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
        self.iroot       = 1  # Electronic state of interest
        self.wf          = wf # File containing the Wavefunction 

        # Input arrays for integrals
        self._bas = []
        self._atm = []
        self._env = [0.0 for i in range(0,20)]

        # Other
        self._pseudonorm = np.array([]) # Pseudonormalization factors

    def update_wf(self, molden_obj):
        """Update wavefunction info from molden file on self.wf"""
        wf_attributes = ["_bas", "_env", "_pseudonorm", "C_mo","moenergies","mooccnos","mospin", "D_ao"]
        for iattr in wf_attributes:
            if(hasattr(molden_obj, iattr)):
                self[iattr] = deepcopy(molden_obj[iattr])
    
    def update_headers(self):
        """"""
        keys = self.__headers.keys()
        for key in keys:
            blist = [key in iline for iline in self.lines]
            if(True in blist):
                # Grab first appearance
                self.__headers[key] = True

    def parse(self):
        """General parser"""
        
        # Parse general info from output file
        for key, val in self.__headers.items():
            if(val and key in self.__parsers.keys()):
                # Call corresponding parser
                self.__parsers[key]()

        # Get wavefunction info, if any
        if(self.wf != ""):
            # Parsing wf data from molden file.
            molden_obj = Molden_parser(self.wf, self.charge, self.mult, self.order, self.basis)
            molden_obj.parse()
            self.update_wf(molden_obj)
        
    def parse_general(self):
        """Parse general molecular properties"""
        prop = {"Charge": "charge",
                "Multiplicity": "mult",
                "Electrons": "nele"}
        for i, iline in enumerate(self.lines):
            if("Charge       =" in iline):
                for j, jline in enumerate(self.lines[i:]):
                    if(len(jline) == 0): break
                    row = jline.split()
#                    if(row[0] in prop.keys()):
                    key = row[0]
                    if(key in prop.keys()):
                        self[prop[key]] = int(row[-1])
                break

    def parse_coords(self):
        """
        Parse general info about the molecule.
        Read directly A.U. entry, which also contains information
        regarding atomic numbers, weights, fragments and ghost atoms.
        """

        for cnt1, iline in enumerate(self.lines):
            if("Geometry (in Angstrom)," in iline):
                for atid, jline in enumerate(self.lines[cnt1 + 4:]):
                    if(len(jline) == 0): break
                    row = jline.split()

                    # Assume input geometry in angstrom, 
                    # then transform in Bohr
                    ixyz  = [ag2bohr * float(k) for k in row[1:4]]
                    if("Gh(" in row[0]):
                        self.ghost.append(atid) #Keep track of ghost atoms
                        ghost_sb = row[0].replace("Gh(", "").replace(")", "")
                        atnum = atom_nums[ghost_sb]
                    else:
                        atnum = atom_nums[row[0]]
                    self.atomcoords.append(ixyz)
                    self._atm.append([atnum, 20 + 4*atid, 1,\
                                      20 + 4*atid + 3, 0, 0]) # add "charge"
                    self._env = self._env + ixyz + [0.0] # add "charge"
                break
        self.atomcoords  = np.array(self.atomcoords).astype("float64")
        # Get atomic numbers
        self._atm    = np.array(self._atm)
        self.atomnos = [iat for iat in self._atm[:,0]]
        self.natoms  = len(self.atomnos)

    def parse_shells(self):
        """
        Parse general info on the basis set. The basis set itself is
        read from the molden file containing the wf.
        """
        prop = {"Number of shells": "nshls",
                "Number of basis functions": "nsph",
                "Number of Cartesian functions": "ncrt"}
        for i, iline in enumerate(self.lines):
            if("Number of shells:" in iline):
                for j, jline in enumerate(self.lines[i:]):
                    if(len(jline) == 0): break
                    row = jline.split(":")
                    key = " ".join(row[0].split())
                    if(key in list(prop.keys())):
                        self[prop[key]] = int(row[-1])
                    # Asses type of basis (cartesian or spherical)
                    if("Spherical Harmonics?" in key):
                        if("".join(row[-1].split()).lower() == "true"):
                            self.basis = "spherical"
                        else:
                            self.basis = "cartesian"
                break

    def get_energies(self):
        """Parse SCF and post-SCF energies. Excited States to be implemented."""
        
        for i, iline in enumerate(self.lines):
            # Parse energies in hartrees, except for self.energies

            if("RKS Final Energy:" in iline or "RHF Final Energy:" in iline):
                etype = "scf_energy"
                self[etype] = float(iline.split()[-1])

            if("=> DF-MP2 Energies <=" in iline):
                etype = "mp2_energy"
                self[etype] = float(self.lines[i + 7].split()[-2])
            
        # Get Energy of the highest level of theory calculation
        lvl = ["scf_energy", "mp2_energy"]
        for ilvl in lvl:
            if(hasattr(self, ilvl)):
                highest = ilvl
        
        ha_ene = self[highest]
        self.haenergies.append(ha_ene)
        self.energies.append(0) # Relative energy in eV
    
    def __getattribute__(self, name):
        return(object.__getattribute__(self, name))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return(object.__getattribute__(self, name))


