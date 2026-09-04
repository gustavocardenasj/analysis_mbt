#########################################################
# MOLDEN WRITER FOR Mol OBJECTS. ONLY SPHERICAL         #
# BASIS FUNCTIONS ARE PRINTED                           #
#########################################################
import numpy as np
#from molden_parser import Mol, gto_sph_norm
from tools_mbt.misc import atom_nums, ag2bohr,\
        ang_mom, gto_sph_norm, atom_names
from collections import OrderedDict  
from copy import deepcopy

# Spherical/cartesian functions
basis_code_sph = {2: "[5D]",
                  3: "[5D7F]",
                  4: "[5D7F9G]"}
basis_code_crt = {2: "[6D]",
                  3: "[6D10F]",
                  4: "[6D10F14G]"}

# PERMUTATIONS MOVED TO permutations.py

# Fortran Format
def fformat(n):
    a = '{:.9E}'.format(float(n))
    e = a.find('E')
    if(a[0] == "-"):
        return '-0.{}{}{}{:02d}'.format(a[1],a[3:e],a[e:e+2],abs(int(a[e+1:])*1+1))
    else:
        return '0.{}{}{}{:02d}'.format(a[0],a[2:e],a[e:e+2],abs(int(a[e+1:])*1+1))

class Write_molden(object):
    """Write molden file. It uses a Mol() object as input"""

    def __init__(self, mol, outfile = "outfile.molden"):
        # Create a deepcopy of the mol input object so as not to alter its attributes
        try:
            self.mol = deepcopy(mol)
        except:
            print('Warning: Mol object cannot be deep copied. \
                  \nA pointer will be used instead. However,\
                  \nchanges will occur in place!')
            self.mol = mol
        self.headers = OrderedDict({"Atoms": "atomcoords", 
                                    "GTO": "_bas", 
                                    "MO": "C_mo",
                                    "FREQ": "freq",
                                    "FR-COORD": "atomcoords",
                                    "FR-NORM-COORD": "nmodes",
                                    "INT": "ir_intens"
                                    })
        self.writers = {"GTO": self.write_gto, 
                        "Atoms": self.write_coords,
                        "MO": self.write_mo,
                        "FREQ": self.write_freq,
                        "FR-COORD": self.write_coords_freq,
                        "FR-NORM-COORD": self.write_nmodes,
                        "INT": self.write_ir_spectrum}

        # Update AO ordering in self.mol to molden
        self.mol.update_order(target = "molden")
        self.outfile = outfile

    def write_output(self):
        """Write molden file"""
        with open(self.outfile, "w") as f:
            for key in self.headers.keys():
                if(hasattr(self.mol, self.headers[key])):
                    self.writers[key](f)
    
    def write_gto(self, f):
        """Write GTOs in molden format. 
           Retrieve info from mol._bas
           and mol._env.
           f = output file (file type)
        """
        ang_mom_num = {ang_mom[key]: key for key in ang_mom.keys()}
        fmt_atm = "{:>3d}{:>2d}\n"
        fmt_bas = "{:>2s}{:>5d}{:>5.2f}\n"
        fmt_prm = "{:>18.10E}{:>18.10E}\n"
        # Get maximum angular momentum
        amom_max = np.max(self.mol._bas[:,1])
        if("spherical" in self.mol.basis):
            f.write(basis_code_sph[amom_max] + "\n") # Spherical basis
        elif("cartesian" in self.mol.basis):
            f.write(basis_code_crt[amom_max] + "\n") # Cartesian basis
        else:
            raise ValueError("{} is not a valid basis type.".format(self.mol.basis))
        f.write("[GTO]\n")

        # Iterate over shells
        curr_atm = self.mol._bas[0][0] + 1
        f.write(fmt_atm.format(curr_atm,0)) # Write first atom
        for cnt, ibas in enumerate(self.mol._bas):
            if((curr_atm - 1) != ibas[0]):
                # Update atom
                curr_atm = ibas[0] + 1
                f.write("\n")
                f.write(fmt_atm.format(curr_atm,0))
            amom  = ang_mom_num[ibas[1]]
            nprim = ibas[2]
            f.write(fmt_bas.format(amom, nprim, 1.00))
            for i in range(nprim):
                exp_id  = ibas[5] + i
                coef_id = ibas[6] + i
                exp     = self.mol._env[exp_id]
                # Remove pseudo-normalization
                coeff   = (1/gto_sph_norm(ibas[1], exp))*self.mol._env[coef_id]
                str_exp   = fformat(exp)
                str_coeff = fformat(coeff)
                iline = "{:>18s}{:>18s}\n".format(str_exp, str_coeff)
                f.write(iline.replace("E", "D"))
        f.write("\n\n")

    def write_coords(self, f):
        """Write coordinates in molden format.
           units: Angstroem
        """
        f.write("[Molden Format]\n")
        f.write("[Atoms] Angs\n")
        atom_list  = [iat for iat in self.mol._atm[:,0]]
        names      = [atom_names[iat] for iat in atom_list]

        fmt = "{:>3s}{:>6d}{:>3d}{:>13.6f}{:>13.6f}{:>13.6f}\n"

        for cnt, icrd in enumerate(self.mol.atomcoords):
            iname = names[cnt]
            inum  = atom_list[cnt]
            f.write(fmt.format(iname, cnt + 1, inum, *(1/ag2bohr)*icrd))

    def write_mo(self, f):
        """Write molecular orbitals. Each row of the C_mo matrix
           corresponds to a column (a MO) in the molden file
        """
        f.write("[MO]\n")
        if(not(hasattr(self.mol, "moenergies"))):
            self.mol.moenergies = [0.0 for i in range(self.mol.nmo)]
        if(not(hasattr(self.mol, "mospin"))):
            self.mol.mospin = ["Alpha" for i in range(self.mol.nmo)]
        if(not(hasattr(self.mol, "mooccnos"))):
            self.mol.mooccnos = [0.0 for i in range(self.mol.nmo)]
        for cnt, imo in enumerate(self.mol.C_mo):
            moene  = self.mol.moenergies[cnt]
            mospin = self.mol.mospin[cnt]
            moocc  = self.mol.mooccnos[cnt]
            # Sym provvisory
            f.write("{:>5s}{:>7s}\n".format("Sym=",str(cnt + 1)+ "a"))
            f.write("{:>5s}{:>11.4f}\n".format("Ene=",moene))
            f.write("{:>6s}{:>6s}\n".format("Spin=",mospin))
            f.write("{:>7s}{:>11.6f}\n".format("Occup=", moocc))
            for cn1, iao in enumerate(imo):
#                f.write("{:>4d}{:>11.6f}\n".format(cn1 + 1, iao))
                # Molcas-generated molden file 
                f.write("{:>4d}{:>19.12f}\n".format(cn1 + 1, iao)) 

    def write_freq(self, f):
        """
        Write vibrational frequencies, if present.
        """
        if(len(self.mol.freq) == 0):
            return
        f.write("[FREQ]\n")
        for cnt, ifreq in enumerate(self.mol.freq):
            f.write("  {:>18.10e}\n".format(ifreq))

    def write_coords_freq(self,f):
        """Write coordinates for normal modes.
           units: Bohr
        """
        if(len(self.mol.freq) == 0):
            return
        f.write("[FR-COORD]\n")
        atom_list  = [iat for iat in self.mol._atm[:,0]]
        names      = [atom_names[iat] for iat in atom_list]

        fmt = "{:<4s}{:>18.10e}{:>18.10e}{:>18.10e}\n"

        for cnt, icrd in enumerate(self.mol.atomcoords):
            iname = names[cnt]
            inum  = atom_list[cnt]
            f.write(fmt.format(iname, *icrd))

    def write_nmodes(self, f):
        """
        Write normal modes of vibration, if present
        """
        if(len(self.mol.nmodes) == 0):
            return
        f.write("[FR-NORM-COORD]\n")
        fmt = "{:>18.10e}{:>18.10e}{:>18.10e}\n"
        for icnt, imode in enumerate(self.mol.nmodes):
            f.write("vibration {}\n".format(icnt+1))
            for jcnt, jatom in enumerate(imode):
                f.write(fmt.format(*jatom))
    
    def write_ir_spectrum(self, f):
        """
        Write IR intensities
        """
        if(len(self.mol.ir_intens) == 0):
            return
        f.write("[INT]\n")
        fmt = "{:>18.10e}\n"
        for cnt, iint in enumerate(self.mol.ir_intens):
            f.write("{:>18.10e}\n".format(iint))

