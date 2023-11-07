#########################################################
# GAMESS DAT WRITER FOR Mol OBJECTS                     #
#########################################################
import numpy as np
from tools_mbt.misc import ang_mom_sp, ang_mom_ct
from tools_mbt.permutations import permute_orbitals
from datetime import datetime
from copy import deepcopy
import os

# Dictionary with Pople shell orderings.
# Permute from a gradually increasing ordering
# in l to a Pople standard ordering
# - e.g.: for C 6-31G* 
# [0, 0, 0, 1, 1, 2] -> [0, 0, 1, 0, 1, 2]
#
# so the permutation is 
# [0, 1, 3, 2, 4, 5]
#
# UPDATE: In practice, what really needs to be swapped are
# the MOs (the columns of mol.C_mo), so that for a given
# atom, if the shells are:
#
# [0, 0, 0, 1, 1, 2],
#
# the corresponding MOs are
#
#  s  s  s px py pz px py pz d...
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14]
#
# And would need to be swapped as:
# 
#  s  s px py pz  s px py pz d...
# [0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14]


POPLE_PERM = {"6-31gd":{1: [0, 1],
                        4: [0, 1, 3, 2, 4, 5],
                        5: [0, 1, 3, 2, 4, 5],
                        6: [0, 1, 3, 2, 4, 5],
                        7: [0, 1, 3, 2, 4, 5],
                        8: [0, 1, 3, 2, 4, 5]
                        }}

POPLE_PERM_MO = {"6-31gd":{1: [0,1],
                           4:[0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14],
                           5:[0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14],
                           6:[0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14],
                           7:[0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14],
                           8:[0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10 ,11, 12, 13, 14]
                           }}
POPLE_LIST = ["6-31g", "6-31gd"]

def write_mo_vector(vec, imo, outfile):
    """
    Write an 1d array on outfile with the .dat
    gamess format. Args:
    vec = The MO to write
    imo = the index of such MO (starts w. 1)
    outfile = output file

    """
    nrows    = int(len(vec)/5)
    mult_5   = len(vec)%5
    mult_100 = imo%100
    fmt1     = "{:>2d}{:>3d}" + 5*"{:15.8E}" + "\n"
    
    for i in range(nrows):
        outfile.write(fmt1.format(mult_100, i+1, *list(vec[5*i:5*i+5])))
    if(mult_5 != 0):
        try:
            ln2  = len(vec[5*(nrows):])
        except:
            ln2 = 1
        fmt2 = "{:>2d}{:>3d}" + ln2*"{:15.8E}" + "\n"
        outfile.write(fmt2.format(mult_100, nrows+1, *list(vec[5*(nrows):])))

class Write_gamess(object):

    def __init__(self, mol, pople = None, outfile = "outfile.dat"):
        """
        Write a GAMESS dat file with a set of molecular 
        orbitals and other initial guess data.
        WARNING: For the time being it only supports cartesian
        basis sets, so that the attribute in mol.basis
        must be cartesian.

        Args: Mol() instance
        """
        # Create a deepcopy of the mol input object so as not to alter its attributes
        try:
            self.mol = deepcopy(mol)
        except:
            print('Warning: Mol object cannot be deep copied. \
                  \nA pointer will be used instead. However,\
                  \nchanges will occur in place!')
            self.mol = mol
        if(pople != None):
            self.pople = pople.lower()
        else:
            self.pople = None
        self.outfile = outfile
        

    def permute_ao(self, order="molcas"):
        perm_id = permute_orbitals(
                                   bas = self.mol._bas, 
                                   order = self.mol.order, 
                                   target = "gamess", 
                                   basis = "cartesian")
        return(perm_id)

    def permute_shells_pople(self):
        """
        Shell reordering in case the input orbitals are not ordered according
        to the Pople basis set ordering.
        In GAMESS, pople shells follow a strict ordering one has to
        be consistent with. For example, for C atom, the 6-31G* basis
        ordering reads:
        
        s,s,px,py,pz,s,px,py,pz,dx2,dy2,dz2,dxy,dxz,dyz
        
        so that the shells are
        
        s,sp,sp,d

        or

        s,s,p,s,p,d

        """
        # Iterate over atoms. Permutations are performed in-place
        # for each atom in mol._bas
        if(self.pople not in POPLE_LIST):
            raise ValueError(self.pople + " is not within the list " + str(POPLE_LIST))

        offset_mo = 0
        for i, iatm in enumerate(np.unique(self.mol._bas[:,0])):
            elem = self.mol._atm[iatm][0]
            perm_elem = POPLE_PERM[self.pople][elem]
            perm_mo   = POPLE_PERM_MO[self.pople][elem]
            # Reindex shells for atom iatm
            bas_per_atm = self.mol._bas[self.mol._bas[:, 0] == iatm]
#            print(self.mol._bas[self.mol._bas[:, 0] == iatm])
            self.mol._bas[self.mol._bas[:, 0] == iatm] =\
                    self.mol._bas[self.mol._bas[:, 0] == iatm][perm_elem]
            # Reindex MOs
            init_idx = offset_mo + np.arange(len(perm_mo))
            fin_idx = offset_mo + np.array(perm_mo) 
            self.mol.C_mo[:,init_idx] = self.mol.C_mo[:,fin_idx]
            offset_mo += len(init_idx)

    def write_output(self):
        """
        Write the MOs saved in the self.mol instance (if
        any) in a GAMESS .dat format file.
        """

        has_mos = hasattr(self.mol, "C_mo")
        if(not has_mos):
            # Get MOs and properties
            self.mol._parser.get_mos()
        # Permute shells so that they follow a Pople 
        # standard, if pople != None
        if(self.pople != None):
            self.permute_shells_pople()
#            self.mol._parser.get_mos()
        # Permute MOs to GAMESS ordering
        perm_ids = self.permute_ao()
        self.C_mo = self.mol.C_mo[:,perm_ids]

        with open(self.outfile, "w") as f:
            f.write(" $VEC\n")
            # Write MO Coefficients
            for imo in range(self.mol.ncrt):
                write_mo_vector(self.C_mo.flatten()[imo*self.mol.ncrt:(imo+1)*self.mol.ncrt], imo+1, f)
            f.write(" $END\n")

