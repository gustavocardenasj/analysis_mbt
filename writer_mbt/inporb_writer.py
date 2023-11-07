#########################################################
# INPORB (Molcas Orbital) WRITER FOR Mol OBJECTS            #
#########################################################
import numpy as np
from tools_mbt.misc import ang_mom_sp
from tools_mbt.permutations import permute_orbitals
from datetime import datetime
from copy import deepcopy
import os


# PERMUTATIONS MOVED TO permutations.py

def write_vector(vec, outfile):
    """Write 1d array on file"""
    nrows  = int(len(vec)/5)
    mult_5 = len(vec)%5
    fmt1   = 5*"{:>22.14E}" + "\n"

    for i in range(nrows):
        outfile.write(fmt1.format(*list(vec[5*i:5*i+5])))
    if(mult_5 != 0):
        try:
            ln2  = len(vec[5*(nrows):])
        except:
            ln2 = 1
        fmt2 = ln2 * "{:>22.14E}" + "\n"
        outfile.write(fmt2.format(*list(vec[5*(nrows):])))

class Write_inporb(object):
    """Write INPORB file. It uses a Mol() object as input.
       Only real spherical harmonics are supported.
    """

    def __init__(self, mol, outfile = "outfile.InpOrb"):
        """Args:
           mol = Mol() object
           order = Ordering of the AOs. Default:
           standard molcas. The AOs still need to
           be swapped to obey the internal ordering
           of Molcas.
        """
        # Create a deepcopy of the mol input object so as not to alter its attributes
        try:
            self.mol = deepcopy(mol)
        except:
            print('Warning: Mol object cannot be deep copied. \
                  \nA pointer will be used instead. However,\
                  \nchanges will occur in place!')
            self.mol = mol
        # Get number of spherical harmonics
        if(self.mol.nsph>0):
            self.nsph = self.mol.nsph
        else:
            self.mol.get_num_sp()
            self.nsph = self.mol.nsph
        self.outfile = outfile
    
    def permute_ao(self, order="molcas"):
        """Permute AO to internal MOLCAS format"""
        perm_id = permute_orbitals(self.mol._bas, self.mol.order, "h5")
        return(perm_id)

    def _permute_ao(self):
        """Permute AOs from standard MOLCAS to internal
           MOLCAS format
        """
        perm_id = []
        cn = 0
        natm = len(self.mol._atm)
        # Iterate over atoms
        for i in range(natm):
            iat = self.mol._bas[self.mol._bas[:,0] == i]
            amoms = np.unique(iat[:,1])
            # Iterate over angular momenta
            for j in amoms:
                shls = len(iat[iat[:,1] == j])
                # Iterate over functions per shell
                for k in range(ang_mom_sp[j]):
                    perm_id += [cn + k + ang_mom_sp[j] * ishl for ishl in range(shls)]
                cn += ang_mom_sp[j] * shls
        return(perm_id)


    def write_inporb(self, order = "molcas"):
        """Transform a molden file into an input orbital 
           Molcas file.
        """
        has_mos = hasattr(self.mol, "C_mo")
        if(not has_mos):
            # Get MOs and properties
            self.mol.get_mos()
        rawdate = datetime.today()
        date    = rawdate.strftime("%a %b %d %H:%M:%S %Y")
        perm_ids = self.permute_ao(order)
        print("len perm_ids = ", len(perm_ids))
        self.C_mo = self.mol.C_mo[:,perm_ids]
        host    = "HOST"
        pid     = "PID"
        
        with open(self.outfile, "w") as f:
            f.write("#INPORB 2.2\n")
            f.write("#INFO\n")
            f.write("* Guess orbitals\n")
            f.write("        0       1       0\n")
            f.write("{:>8d}\n".format(self.nsph))
            f.write("{:>8d}\n".format(self.nsph))
            f.write("*BC:HOST {} PID {} DATE {}\n".format(host, pid, date))
            f.write("#ORB\n")
            # Write MO Coefficients
            for imo in range(self.nsph):
                f.write("* ORBITAL    1    {:d}\n".format(imo + 1))
                write_vector(self.C_mo.flatten()[imo*self.nsph:(imo+1)*self.nsph], f)
            # Write occupations
            f.write("#OCC\n")
            if(hasattr(self.mol, "mooccnos")):
                occs = self.mol.mooccnos 
            else:
                occs = np.zeros((self.nsph))
            write_vector(occs, f)
            
