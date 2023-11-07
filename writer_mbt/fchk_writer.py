#########################################################
# FCHK WRITER FOR Mol OBJECTS. ONLY SPHERICAL           
# BASIS FUNCTIONS ARE PRINTED                           #
#########################################################
import numpy as np
from tools_mbt.misc import atom_nums, ag2bohr,\
        ang_mom, gto_sph_norm, atom_names,\
        weights
from collections import OrderedDict  
from copy import deepcopy

# Spherical functions
basis_code = {2:"[5D]",
              3: "[5D7F]",
              4: "[5D7F9G]"}

# Define formats
fmt_int   = "{:<40s}   I     {:>12d}\n"
fmt_real  = "{:<40s}   R     {:<22.15E}\n"
iarr_head = "{:<40s}   I   N={:>12d}\n"
rarr_head = "{:<40s}   R   N={:>12d}\n"

def write_lin_arr(f, arr, tp = int):
    """Write a linearized array on file f, with a 
       maximum number of ncols 
       columns.
    """
    arr      = np.array(arr).flatten()
    tp       = str(arr.dtype)
    size     = len(arr)
    if(tp == "int64"):
        if(size<6):
            ncols = size
        else:
            ncols    = 6
        fmt      = ncols * "{:>12d}" + "\n"
        fmt_last = size%ncols * "{:>12d}" + "\n"
    elif(tp == "float64"):
        if(size<5):
            ncols = size
        else:
            ncols    = 5
        fmt      = ncols * "{:>16.8E}" + "\n"
        fmt_last = size%ncols * "{:>16.8E}" + "\n"
    else:
        raise(ValueError(str(tp) + "Does not belong to a valid type"))

    nrows    = int(size/ncols)          # Floor
    if(size<ncols):
        # Write a single row
        f.write(fmt_last.format(*arr))
    else:
        for i in range(nrows):
            f.write(fmt.format(*arr[i*ncols:(i+1)*ncols]))
    if(size%ncols > 0):
        f.write(fmt_last.format(*arr[nrows*ncols:]))
        

class Write_fchk(object):
    """Write fchk file. It uses a Mol() object as 
       input. The mol object has the MO coefficients 
       with a molden ordering.
    """

    def __init__(self, mol, outfile = "outfile.fchk"):
        # Create a deepcopy of the mol input object so as not to alter its attributes
        self.mol = deepcopy(mol)
        self.headers = OrderedDict({"Atoms": \
                "atomcoords", "GTO": "_bas", \
                "MO": "C_mo",
                "Density": "D_ao"})
        self.writers = {"GTO": self.write_gto, 
                        "Atoms": self.write_mol_info,
                        "MO": self.write_mo,
                        "Density": self.write_density}
        
        # Update AO ordering in self.mol to molden,
        # which is the same as that of the fchk file
        self.mol.update_order(target = "molden")
        self.outfile = outfile

    def write_output(self):
        """Write fchk file"""
        with open(self.outfile, "w") as f:
            self.write_title(f)
            for key in self.headers.keys():
                if(hasattr(self.mol, self.headers[key])):
                    self.writers[key](f)
#            self.write_mol_info(f)
#            self.write_gto(f)
#            self.write_mo(f)

    def write_title(self, f):
        """Write default title"""
        fmt = "{:<72s}\n{:<10s}{:<30s}{:<30s}\n"
        if(self.mol.density != "None"):
            calc_type = self.mol.density
        else:
            calc_type = "HF"
#        f.write(fmt.format("title", "SP", "HF", "gen"))
        f.write(fmt.format("title", "SP", calc_type, "gen"))

    def write_mol_info(self, f):
        """Write general molecular information."""
        if(not(hasattr(self.mol, "nele"))):
            ele  = np.sum(self.mol.atomnos) - self.mol.charge
        else:
            ele = self.mol.nele
        aele = int(np.ceil(ele/2))  # Alpha electrons
        bele = int(np.floor(ele/2)) # Beta electrons 

        f.write(fmt_int.format("Number of atoms", int(self.mol.natoms)))
        f.write(fmt_int.format("Charge", int(self.mol.charge)))
        f.write(fmt_int.format("Multiplicity", int(self.mol.mult)))
        f.write(fmt_int.format("Number of electrons", int(ele)))
        f.write(fmt_int.format("Number of alpha electrons", int(aele)))
        f.write(fmt_int.format("Number of beta electrons", int(bele)))
        f.write(fmt_int.format("Number of basis functions", self.mol.nsph))
        f.write(fmt_int.format("Number of independent functions", int(self.mol.nsph)))
        f.write(fmt_int.format("Number of point charges in /Mol/", 0))
        f.write(fmt_int.format("Number of translation vectors", 0))

        # Print atomic numbers
        int_atomnum  = np.array(self.mol.atomnos).astype("int")
        real_atomnum = np.array(self.mol.atomnos).astype("float64")
        # Account for ghost atoms
        if(hasattr(self.mol, "ghost")):
            if(len(self.mol.ghost) > 0):
                real_atomnum[self.mol.ghost] = 0.0000
        print("INT ATOMNUM = ", int_atomnum)
        print("REAL ATOMNUM = ", real_atomnum)
        f.write(iarr_head.format("Atomic numbers", self.mol.natoms))
        write_lin_arr(f, int_atomnum)
        f.write(rarr_head.format("Nuclear charges", self.mol.natoms))
        write_lin_arr(f, real_atomnum)

        # Print cartesian coords (in bohr)
        f.write(rarr_head.format("Current cartesian coordinates", len(self.mol.atomcoords.flatten())))
        write_lin_arr(f, self.mol.atomcoords)

        # Print atomic weights
#        int_atomw  = np.array([weights[i] for i in self.mol.atomnos]).astype("int")
#        real_atomw = int_atomw.astype("float64")
        int_atomw  = [round(weights[i]) for i in self.mol.atomnos]
        real_atomw = [weights[i] for i in self.mol.atomnos]
        f.write(iarr_head.format("Integer atomic weights", self.mol.natoms))
        write_lin_arr(f, int_atomw)
        f.write(rarr_head.format("Real atomic weights", self.mol.natoms))
        write_lin_arr(f, real_atomw)

        f.write(rarr_head.format("Nuclear ZNuc", self.mol.natoms))
        write_lin_arr(f, real_atomnum)

    def write_gto(self, f):
        """Write general basis set info
           Retrieve info from mol._bas
           and mol._env.
        """
        # Print general basis set info
        ncont = len(self.mol._bas)
        nprim = np.sum(self.mol._bas[:,2])
        lmax  = np.max(self.mol._bas[:,1])
        cmax  = np.max(self.mol._bas[:,2])
        shl_t = [] # Define shell types
        for cnt, ish in enumerate(self.mol._bas[:,1]):
            if(ish in [2,3]):
                shl_t.append(-1*ish)
            else:
                shl_t.append(ish)
        shl_t = np.array(shl_t)

        f.write(fmt_int.format("Number of contracted shells", ncont))
        f.write(fmt_int.format("Number of primitive shells", nprim))
        f.write(fmt_int.format("Pure/Cartesian d shells", 0))
        f.write(fmt_int.format("Pure/Cartesian f shells", 0))
        f.write(fmt_int.format("Highest angular momentum", lmax))
        f.write(fmt_int.format("Largest degree of contraction", cmax))
        f.write(iarr_head.format("Shell types", ncont))
        write_lin_arr(f, shl_t)
        f.write(iarr_head.format("Number of primitives per shell", ncont))
        write_lin_arr(f, self.mol._bas[:,2])
        f.write(iarr_head.format("Shell to atom map", ncont))
        write_lin_arr(f, self.mol._bas[:,0] + 1)

        # Write exponents and coefficients
        expn, coef = self.get_primitives() # Get exponents and coefficients
        f.write(rarr_head.format("Primitive exponents", nprim))
        write_lin_arr(f, expn)
        f.write(rarr_head.format("Contraction coefficients", nprim))
        write_lin_arr(f, coef)

        # Write coordinates per contraction
        f.write(rarr_head.format("Coordinates of each shell", 3*ncont))
        crds = self.get_coordinates()
        write_lin_arr(f, crds)

        # Write various energies (TO BE UPDATED FOR EXCITED STATES)
        if(hasattr(self.mol, "scf_energy")):
            f.write(fmt_real.format("SCF Energy", self.mol.scf_energy))
        if(hasattr(self.mol, "mp2_energy")):
            f.write(fmt_real.format("MP2 Energy", self.mol.mp2_energy))
        if(hasattr(self.mol, "cc_energy")):
            f.write(fmt_real.format("Cluster Energy", self.mol.cc_energy))
        if(hasattr(self.mol, "haenergies")):
            f.write(fmt_real.format("Total Energy", self.mol.haenergies[0]))
    
    def write_mo(self, f):
        """Write molecular orbitals in C_mo as a 
           linearized array.
        """
        # Write molecular orbitals (and relative properties)
        nmos = len(self.mol.moenergies)
        f.write(rarr_head.format("Alpha Orbital Energies", nmos))
        write_lin_arr(f, self.mol.moenergies)
        f.write(rarr_head.format("Alpha MO coefficients", len(self.mol.C_mo.flatten())))
        write_lin_arr(f, self.mol.C_mo)
    
    def write_density(self, f):
        """
        Write density matrix
        """
        # Lower triangular matrix only, since D is symmetric
        nrows  = len(self.mol.D_ao)
        id_upt = np.tril_indices(nrows) # indices lower triangular mat.
        D_flat = self.mol.D_ao[id_upt]  # Flattened symmetric array
        nelem  = nrows*(nrows-1)/2 + nrows
        density_parser = self.mol._density_parser[self.mol.density.upper()]
        f.write(rarr_head.format(density_parser, int(nelem)))
        write_lin_arr(f, D_flat)

    def get_primitives(self):
        """Get 2 linearized arrays of primitive exponents and 
           coefficients
        """
        expn, coef = np.array([]), np.array([])
        for ibas in self.mol._bas:
            nprm, id_expn, id_coef = ibas[2], ibas[5], ibas[6]
            expn = np.concatenate((expn, self.mol._env[id_expn: id_expn + nprm]))
            coef = np.concatenate((coef, self.mol._env[id_coef: id_coef + nprm]))
        # Remove pseudo-normalization
        coef = (1/self.mol._pseudonorm) * coef
        return(expn, coef)

    def get_coordinates(self):
        """Get a linearized array of the cartesian coordinates for each
           contracted gaussian
        """
        crds = np.array([])
        for cnt, ibas in enumerate(self.mol._bas):
            id_atm = ibas[0]
            crds = np.concatenate((crds, self.mol.atomcoords[id_atm]))
        return(crds)

if(__name__ == "__main__"):
    from molden_parser import Mol

    mol = Mol("test.molden")
    mol.parse()
    mol.get_mos()

    writer = Write_fchk(mol)
    writer.write_output()






