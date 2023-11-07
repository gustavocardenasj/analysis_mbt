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


class Molden_parser(object):

    def __init__(self, infile, charge = 0, mult = 1, order = "pyscf", basis = 'spherical'):
        """Molden parser class"""
        # Headers, to evidence which
        # data can be retrieved
        self.__headers = OrderedDict({"ATOMS": False,
            "[GTO]": False,
            "[MO]":  False,
            "[FORCES]": False,
            "[FREQ]": False,
            "[FR-NORM-COORD]": False
            })

        # Lines from infile
        with open(infile, "r") as f:
            self.lines = f.read().splitlines()

        # Update self.headers, according to the
        # headers in self.lines
        self.update_headers()

#        self.parsers = {"GTO": self.parse_shells,
#                        "ATOMS": self.parse_coords,
#                        "MO": self.get_mo}  
        self.__parsers = {"[GTO]": self.parse_shells,
                          "[MO]" : self.get_mos,
                          "ATOMS": self.parse_coords,
                          "[FREQ]": self.parse_frequencies,
                          "[FR-NORM-COORD]": self.parse_nmodes}
        
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
        # Assess whether this is an orca molden file
        # (in which primitives are normalized)
        if("Molden file created by orca_2mkl" in self.lines[2]):
            self.orca_molden = True
        else:
            self.orca_molden = False


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
            blist = [key in iline.upper() for iline in self.lines]
            if(True in blist):
                # Grab first appearance
                self.__headers[key] = True
    
    def parse_frequencies(self):
        """Parse frequencies in cm-1"""
        for cnt1, iline in enumerate(self.lines):
            if(iline[1:5].lower() == "freq"):
                for cnt2, jline in enumerate(self.lines[cnt1+1:]):
                    if("[" in jline or len(jline) == 0):
                        break
                    self.freq.append(float(jline))
                break
        
        # Transform into a 1-d array
        self.freq = np.array(self.freq)

    def parse_nmodes(self):
        """Parse normal modes"""
        for cnt1, iline in enumerate(self.lines):
            if(iline[1:14].lower() == "fr-norm-coord"):
                # Iterate over normal modes
                for cnt2, jline in enumerate(self.lines[cnt1 + 1:]):
                    if("[" in jline or len(jline) == 0):
                        break
                    if("vibration" in jline):
                        # Iterate over xyz coords
                        xyz_buff = [] # Buffer for xyz coords
                        for cnt3, kline in enumerate(self.lines[cnt1 + 1 + cnt2 + 1:]):
                            if(("vibration" in kline) or ("[" in kline) or (len(kline) == 0)):
                                break
                            ixyz = [float(icrd) for icrd in kline.split()[-3:]]
                            xyz_buff.append(ixyz)
                    # Update nmodes list
                    self.nmodes.append(xyz_buff)
                break
        # Transform nmodes list into a rank 3 array
        self.nmodes = np.array(self.nmodes)

    def parse_coords(self):
        """Parse general info about the molecule"""
        for cnt1, iline in enumerate(self.lines):
            if(iline[1:6].lower() == "atoms"):
                if("angs" in iline.lower()):
                    fact = ag2bohr
                elif("bohr" in iline.lower()):
                    fact = 1
                else:
                    fact = 1
                for atid, jline in enumerate(self.lines[cnt1 + 1:]):
                    if(("[" in jline) or (len(jline) == 0)):
                        break
                    ixyz  = jline.split()[-3:] # Last 3 columns
                    ixyz  = [fact * float(i) for i in ixyz]
                    try:
                        atnum = atom_nums[jline.split()[0]] 
                    except:
                        atnum = atom_nums[jline.split()[0][0]] 
                    self.atomcoords.append(ixyz)
                    self._atm.append([atnum, 20 + 4*atid, 1,\
                                      20 + 4*atid + 3, 0, 0]) # add "charge"
                    self._env = self._env + ixyz + [0.0] # add "charge"
                break
        self.atomcoords = np.array(self.atomcoords).astype("float64")
        # Get atomic numbers
        self._atm    = np.array(self._atm)
        self.atomnos = [iat for iat in self._atm[:,0]]
        self.natoms  = len(self.atomnos)

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
        if(self.orca_molden):
            # Call the parser relative to the orca molden format
            self.parse_shells_orca()
            return
        prm       = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]
        iatom     = 0 # Atom index
        iprim     = 0 # Primitive index
        skiplines = 0 # Skip primitive lines after reading them 
    #    offset    = 0 # Offset of a given contraction within a shell
        for cnt1, iline in enumerate(self.lines):
            # Pre-processing: read all basis functions
            if(iline[1:4].lower() == "gto"):
                for cnt2, jline in enumerate(self.lines[cnt1 + 1:]):
                    row = jline.split()
                    if(("[" in jline)):
                        break
                    elif(skiplines > 0):
                        skiplines -= 1
                        pass
                    elif(len(row) == 0):
                        pass
                    elif((len(row) >= 1) and (row[0].isdigit())):
                        iatom = int(row[0]) - 1
                    elif(row[0].lower() in ang_mom.keys()):
                        # Construct reference arrays for ith sell
                        Nprim = int(row[1])
                        amom  =  ang_mom[row[0].lower()]
                        self.ncrt  += ang_mom_ct[amom]
                        # Parse all primitives. sp case should also be considered here
                        currid = cnt1 + 1 + cnt2 + 1
                        cf     = []
                        expn   = []
                        for cnt3, kline in enumerate(self.lines[currid: currid + Nprim]):
                            p1, c1 = kline.replace("D","E").split()
                            p1, c1 = float(p1), float(c1)
                            expn.append(p1)
                            # Multiply gto_sph_norm by coeff
                            cf.append(gto_sph_norm(amom,p1)*c1)
                            self._pseudonorm = np.concatenate((\
                                    self._pseudonorm,[gto_sph_norm(amom,p1)]))
                        if(expn + cf not in prm):
                            prm.append(expn + cf)
                            self._env = self._env + expn + cf
                            idexp     = len(self._env) - 2*Nprim
                            idcf      = len(self._env) - Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        else:
                            natom = len(self.atomcoords)
                            id_prm = prm.index(expn + cf)
                            flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                            idexp = 20 + 4*natom + len(flat_prm)
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        skiplines = Nprim
                    
                    elif(row[0].lower() == "sp"): # Consider sp case
                        Nprim = int(row[1])
                        self.ncrt += 4 # + 1s + 3p
                        currid = cnt1 + 1 + cnt2 + 1
                        amom1 =  ang_mom[row[0][0].lower()]
                        amom2 =  ang_mom[row[0][1].lower()]
                        cf1    = []
                        cf2    = []
                        expn   = []
                        # Buffers for pseudonorm
                        buf1   = []
                        buf2   = []
                        for cnt3, kline in enumerate(self.lines[currid: currid + Nprim]):
                            p1, c1, c2 = kline.replace("D","E").split()
                            p1, c1, c2 = float(p1), float(c1), float(c2)
                            expn.append(p1)
                            # Multiply gto_sph_norm by coeff
                            cf1.append(gto_sph_norm(amom1,p1)*c1)
                            cf2.append(gto_sph_norm(amom2,p1)*c2)
                            buf1.append(gto_sph_norm(amom1,p1)*c1)
                            buf2.append(gto_sph_norm(amom2,p1)*c2)
                        self._pseudonorm = np.concatenate((\
                                self._pseudonorm, buf1 + buf2))
                        iprm = expn + cf1 + expn + cf2 # expn and cf of s and p
                        if(iprm not in prm):
                            prm.append(iprm)
                            self._env = self._env + iprm
                            # Add s shell
                            amom =  ang_mom["s"]
                            idexp     = len(self._env) - 4*Nprim
                            idcf      = len(self._env) - 3*Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                            # Add p shell
                            amom =  ang_mom["p"]
                            idexp     = len(self._env) - 2*Nprim
                            idcf      = len(self._env) - Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        else:
                            # Add s shell
                            natom = len(self.atomcoords)
                            amom =  ang_mom["s"]
                            id_prm = prm.index(iprm)
                            flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                            idexp = 20 + 4*natom + len(flat_prm)
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                            # Add p shell
                            amom =  ang_mom["p"]
                            idexp = idexp + 2*Nprim
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        skiplines = Nprim
    
        # Get number of basis functions
        self.get_num_ct()
        self.get_num_sp()

        # Re-define _bas and _env as numpy arrays
        self._bas = np.array(self._bas)
        self._env = np.array(self._env)

    def parse_shells_orca(self):
        """
        Same as self.parse_shells, but without pseudo-normalization
        """
        prm       = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]
        iatom     = 0 # Atom index
        iprim     = 0 # Primitive index
        skiplines = 0 # Skip primitive lines after reading them 
    #    offset    = 0 # Offset of a given contraction within a shell
        for cnt1, iline in enumerate(self.lines):
            # Pre-processing: read all basis functions
            if(iline[1:4].lower() == "gto"):
                for cnt2, jline in enumerate(self.lines[cnt1 + 1:]):
                    row = jline.split()
                    if(("[" in jline)):
                        break
                    elif(skiplines > 0):
                        skiplines -= 1
                        pass
                    elif(len(row) == 0):
                        pass
                    elif((len(row) >= 1) and (row[0].isdigit())):
                        iatom = int(row[0]) - 1
                    elif(row[0].lower() in ang_mom.keys()):
                        # Construct reference arrays for ith sell
                        Nprim = int(row[1])
                        amom  =  ang_mom[row[0].lower()]
                        self.ncrt  += ang_mom_ct[amom]
                        # Parse all primitives. sp case should also be considered here
                        currid = cnt1 + 1 + cnt2 + 1
                        cf     = []
                        expn   = []
                        for cnt3, kline in enumerate(self.lines[currid: currid + Nprim]):
                            p1, c1 = kline.replace("D","E").split()
                            p1, c1 = float(p1), float(c1)
                            expn.append(p1)
                            # Adapt coefficients to PySCF internal
                            # pseudonormalization 
#                            prec1  = (1/gto_sph_norm_orca(amom, p1)) *  gto_sph_norm(amom,p1)
                            prec1  = gto_sph_norm(amom,p1)
                            orcf1  = (1/gto_sph_norm_orca(amom, p1)) *  prec1
#                            cf.append(prec1 * c1)
                            cf.append(orcf1 * c1)
                            self._pseudonorm = np.concatenate((\
                                    self._pseudonorm,[prec1]))
#                            cf.append(c1)
                        if(expn + cf not in prm):
                            prm.append(expn + cf)
                            self._env = self._env + expn + cf
                            idexp     = len(self._env) - 2*Nprim
                            idcf      = len(self._env) - Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        else:
                            natom = len(self.atomcoords)
                            id_prm = prm.index(expn + cf)
                            flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                            idexp = 20 + 4*natom + len(flat_prm)
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        skiplines = Nprim
                    
                    elif(row[0].lower() == "sp"): # Consider sp case
                        Nprim = int(row[1])
                        self.ncrt += 4 # + 1s + 3p
                        currid = cnt1 + 1 + cnt2 + 1
                        amom1 =  ang_mom[row[0][0].lower()]
                        amom2 =  ang_mom[row[0][1].lower()]
                        cf1    = []
                        cf2    = []
                        expn   = []
                        # Buffers for pseudonorm
                        buf1   = []
                        buf2   = []
                        for cnt3, kline in enumerate(self.lines[currid: currid + Nprim]):
                            p1, c1, c2 = kline.replace("D","E").split()
                            p1, c1, c2 = float(p1), float(c1), float(c2)
                            expn.append(p1)
                            # Adapt coefficients to PySCF internal
                            # pseudonormalization 
#                            prec1  = (1/gto_sph_norm_orca(amom1, p1)) *  gto_sph_norm(amom1,p1)
#                            prec2  = (1/gto_sph_norm_orca(amom1, p2)) *  gto_sph_norm(amom1,p2)
                            prec1  =  gto_sph_norm(amom1,p1)
                            prec2  =  gto_sph_norm(amom1,p2)
                            orcf1  = (1/gto_sph_norm_orca(amom1, p1)) * prec1
                            orcf2  = (1/gto_sph_norm_orca(amom1, p2)) * prec2
#                            cf1.append(prec1 * c1)
#                            cf2.append(prec2 * c2)
                            cf1.append(orcf1 * c1)
                            cf2.append(orcf2 * c2)
                            buf1.append(prec1 * c1)
                            buf2.append(prec2 * c2)
                        self._pseudonorm = np.concatenate((\
                                self._pseudonorm, buf1 + buf2))
#                            cf1.append(c1)
#                            cf2.append(c2)
                        iprm = expn + cf1 + expn + cf2 # expn and cf of s and p
                        if(iprm not in prm):
                            prm.append(iprm)
                            self._env = self._env + iprm
                            # Add s shell
                            amom =  ang_mom["s"]
                            idexp     = len(self._env) - 4*Nprim
                            idcf      = len(self._env) - 3*Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                            # Add p shell
                            amom =  ang_mom["p"]
                            idexp     = len(self._env) - 2*Nprim
                            idcf      = len(self._env) - Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        else:
                            # Add s shell
                            natom = len(self.atomcoords)
                            amom =  ang_mom["s"]
                            id_prm = prm.index(iprm)
                            flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                            idexp = 20 + 4*natom + len(flat_prm)
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                            # Add p shell
                            amom =  ang_mom["p"]
                            idexp = idexp + 2*Nprim
                            idcf  = idexp + Nprim
                            self._bas.append([iatom, amom, Nprim, 
                                              1, 0, idexp, idcf, 0])
                        skiplines = Nprim
    
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
    
    def get_mo_prop(self):
        """Get MO properties such as Energies,
           spin and occupation numbers
        """
        self.moenergies = []
        self.mooccnos   = [] # MO occupation numbers
        self.mospin     = []
        for cnt, iline in enumerate(self.lines):
            row = iline.split()
            if("Ene=" in iline):
                self.moenergies.append(float(row[1]))
            elif("Spin=" in iline):
                self.mospin.append(row[1])
            elif("Occup=" in iline):
                self.mooccnos.append(float(row[1]))
            else:
                pass
        self.moenergies = np.array(self.moenergies)
        self.mooccnos   = np.array(self.mooccnos)
        self.mospin     = np.array(self.mospin)

        # Infere number of electrons and total charge
        nele = np.sum(self.mooccnos)
        Zchg = np.sum(self.atomnos)
        if(nele>0):
            self.nele = round(nele)
            self.charge = float(round(Zchg - nele))
    
    def get_all_mo(self, unrestr = False):
        """For now, assume that N_sph has been parsed. In case of unrestricted solutions,
        this method initializes two matrices: C_aplha and C_beta."""
        if(unrestr == True):
            self._get_mo_unr(parser = "Spin= Alpha")
            self._get_mo_unr(parser = "Spin= Beta")
        else:
            self._get_mo()

    def _get_mo(self, parser = "Spin= Alpha"):
        # Parse
        row_cnt = 0
        for cnt1, lin1 in enumerate(self.lines):
#            if(("MO" in lin1)): # or (lin1 == "[MO]")):
            if(("MO" in lin1) and ("MOLDEN" not in lin1)): # or (lin1 == "[MO]")):
                for cnt2, lin2 in enumerate(self.lines[cnt1:]):
                    if("SCF" in lin2): break
                    if(parser in lin2):
                        for cnt3, lin3 in enumerate(self.lines[cnt1 + cnt2 +2:]):
                            if(("Sym=" in lin3) or ( "Ene=" in lin3) or ( "SCF" in lin3) or ("Occup=" in lin3)): break
                            self.C_mo[row_cnt][cnt3] = float(lin3.split()[1])
                        row_cnt += 1
                break
        
        # Permute AOs so as to follow the self.order
        # ordering
        perm_id = permute_orbitals(self._bas,\
                order = "molden", target = self.order, basis = self.basis)
        self.C_mo = self.C_mo[:,perm_id]

        # Update f and g orbitals in the case of orca_molden
        if(self.orca_molden and self.basis == "spherical"):
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

    def _get_mo_unr(self, parser = "Spin= Alpha"):
        spin = parser.replace("Spin= ", "").lower()
        row_cnt = 0
        for cnt1, lin1 in enumerate(self.lines):
#            if(("MO" in lin1)): # or (lin1 == "[MO]")):
            if(("MO" in lin1) and ("MOLDEN" not in lin1)): # or (lin1 == "[MO]")):
                for cnt2, lin2 in enumerate(self.lines[cnt1:]):
                    if("SCF" in lin2): break
                    if(parser in lin2):
                        for cnt3, lin3 in enumerate(self.lines[cnt1 + cnt2 +2:]):
                            if(("Sym=" in lin3) or ( "Ene=" in lin3) or ( "SCF" in lin3)): break
                            self.C_mo[spin][row_cnt][cnt3] = float(lin3.split()[1])
                        row_cnt += 1
        
        # Permute AOs so as to follow the self.order
        # ordering
        perm_id = permute_orbitals(self._bas,\
                order = "molden", target = self.order)
        self.C_mo = self.C_mo[spin][:,perm_id]
    
    def get_mos(self, unrestr = False):
        """Allocate MO arrays and retrieve corresponding values"""
        # Get MOs
        self.allocate_mo(unrestr)
        self.get_all_mo(unrestr)
        #Get properties
        self.get_mo_prop()

    def __getattribute__(self, name):
        return(object.__getattribute__(self, name))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return(object.__getattribute__(self, name))

if(__name__ == "__main__"):
    import sys

    infile = sys.argv[1]
    mol = Molden_parser(infile)
    mol.parse()

