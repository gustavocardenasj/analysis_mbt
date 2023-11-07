#################################################################
# VARIOUS PERMUTATION HANDLERS, ESPECIALLY ATOMIC               #
# ORBITALS. ONLY SPHERICAL AND CARTESIAN FUNCTIONS SUPPORTED    #
#################################################################

import numpy as np
from tools_mbt.misc import atom_nums, ag2bohr, ang_mom_sp,\
        ang_mom, gto_sph_norm, atom_names
from collections import OrderedDict  

# Permutaions among various AO orderings
def permute_orbitals(bas, order="molcas", target="molden", basis = 'spherical'):
    """Read a bas list (Mol attribute) and permmute orbital 
       indices among molden (Gausian), pyscf, molcas 
       (output file), h5 (molcas inporb and h5)
       
       Return a list of indices with the desired ordering.
       NOTICE: These are in-shell permutations, and do 
               not permute basis functions among different
               shells.

       ORDERING DESCRIPTION:
       
       ############### ORDERING FOR REAL SPHERICAL FUNCTIONS ############### 
    
       # MOLDEN (gaussian) ordering:
       # P: X, Y ,Z (+1, -1, 0)
       # D: DZ2, DXZ, DYZ, DY2, DXY (0, +1, -1, +2, -2)
       # F: (0, +1, -1, +2, -2, +3, -3)
   
       # PYSCF ORDERING
       # P: Y, Z, X (-1, 0, +1)
       # D: DXY, DYZ, DZ2, DXZ, DY2 (-2, -1, 0, +1, +2)
       # F: (-3, -2, -1, 0, 1, 2, 3)

       # ORCA output ORDERING (same as Gaussian, but p)
       # P: Z, X, Y (0, +1, -1)
       # D: DZ2, DXZ, DYZ, DY2, DXY (0, +1, -1, +2, -2)
       # F: (0, +1, -1, +2, -2, +3, -3)

       # HORTON ORDERING (same as Gaussian, but p)
       # P: Z, X, Y
       # D: DZ2, DXZ, DYZ, DY2, DXY (0, +1, -1, +2, -2)
       # F: FZ3, FXZ2, FYZ2, FX2Z, FXYZ, FX3, FX2Y
       #    (0, +1, -1, +2, -2, +3, -3)
   
       # MOLCAS ORDERING 1: (here molcas - output file)
       # SHELLS ARE GROUPED
       # P: X, Y ,Z (+1, -1, 0)
       # D: DXY, DYZ, DZ2, DXZ, DY2 (-2, -1, 0, +1, +2)
       # F: (-3, -2, -1, 0, 1, 2, 3)
       
       # MOLCAS ORDERING 2: HDF5 (here h5) 
       # Same as molcas but regrouping shells 
       # (see h5_to_pyscf)
       # P: X, Y ,Z (+1, -1, 0)
       # D: DXY, DYZ, DZ2, DXZ, DY2 (-2, -1, 0, +1, +2)
       # F: (-3, -2, -1, 0, 1, 2, 3)
       
       # MOLCAS ORDERING 3: MULLIKEN CHG (output file)
       # P: X, Y ,Z (+1, -1, 0)
       # D: DXY, DYZ, DZ2, DXZ, DY2 (-2, -1, 0, +1, +2)
       # D: DY2, DXZ, DZ2, DYZ, DXY (+2, +1, 0, -1, -2)
       # F: (+3, +2, +1, 0, -1, -2, -3)

      
       ############### ORDERING FOR REAL CARTESIAN FUNCTIONS ###############
       
       # MOLDEN (gaussian) ordering:
       # P: X, Y, Z 
       # D: DX2, DY2, DZ2, DXY, DXZ, DYZ
       # F: FX3, FY3, FZ3, FXYY, FXXY, FXXZ, FXZZ, FYZZ, FYYZ, FXYZ  
       #    0  , 1  , 2  , 3   , 4   , 5   , 6   , 7   , 8   , 9   

       # PYSCF ordering:
       # P: Y, Z, X
       # D: DX2, DXY, DXZ, DY2, DYZ, DZ2
       # F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
       #    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9

       # Horton ordering:
       # P: Z, X, Y
       # D: DX2, DXY, DXZ, DY2, DYZ, DZ2 (same as PYSCF)
       # F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3 (same as PYSCF)
       #    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9

       # MOLCAS: Always prints spherical functions, except for Pople
       # Notice that here (as in h5) the shells of equal magnetic moment
       # are regrouped
       # P: X, Y, Z
       # D: DX2, DXY, DXZ, DY2, DYZ, DZ2

       # GAMESS ORDERING
       # P: X, Y, Z
       # D: DX2, DY2, DZ2, DXY, DXZ, DYZ
       # F: FX3, FY3, FZ3, FXXY, FXXZ, FXYY, FYYZ, FXZZ, FYZZ, FXYZ
       #    0  , 1  , 2  , 3   , 4   , 5   , 6   , 7   , 8   , 9

    """

    count = 0
    idlist = []

    # PERMUTATIONS FOR SPHERICAL FUNCTIONS
    if(basis == "spherical"):
        opt1 = ("molcas" in order.lower()) and ("molden" in target.lower())
        opt2 = ("pyscf" in order.lower()) and ("molden" in target.lower())
#        opt2 = "molden" in order.lower() & "molcas" in target.lower()
        opt3 = ("molden" in order.lower()) and ("pyscf" in target.lower())
        opt4 = ("molden" in order.lower()) and ("molcas" in target.lower())
        opt5 = ("pyscf" in order.lower()) and ("molcas" in target.lower())
        opt6 = ("pyscf" in order.lower()) and ("h5" in target.lower())
        opt7 = ("pyscf" in order.lower()) and ("horton" in target.lower())
        opt8 = ("orca" in order.lower()) and ("pyscf" in target.lower())
        equal= (order.lower() == target.lower())
        if(opt1):
            p_idx = [0, 1, 2]
            d_idx = [2, 3, 1, 4, 0]
            f_idx = [3, 4, 2, 5, 1, 6, 0]
        elif(opt2):
            p_idx = [2, 0, 1]
            d_idx = [2, 3, 1, 4, 0]
            f_idx = [3, 4, 2, 5, 1, 6, 0]
        elif(opt3):
            p_idx = [1, 2, 0]
            d_idx = [4, 2, 0, 1, 3]
            f_idx = [6, 4, 2, 0, 1, 3, 5]
        elif(opt4):
            p_idx = [0, 1, 2]
            d_idx = [4, 2, 0, 1, 3]
            f_idx = [6, 4, 2, 0, 1, 3, 5]
        elif(opt5):
            # pyscf and molcas differ on the ordering
            # of px, py, pz
            p_idx = [2, 0, 1]
            d_idx = [0, 1, 2, 3, 4]
            f_idx = [0, 1, 2, 3, 4, 5, 6]
        elif(opt6):
            return(pyscf_to_h5(bas))
        elif(opt7):
            p_idx = [1, 2, 0]
            d_idx = [2, 3, 1, 4, 0]
            f_idx = [6, 4, 2, 0, 1, 3, 5]
        elif(opt8):
            p_idx = [2, 0, 1] # z,x,y -> y,z,x
            d_idx = [4, 2, 0, 1, 3]
            f_idx = [6, 4, 2, 0, 1, 3, 5]
        elif(equal):
            p_idx = [0, 1, 2]
            d_idx = [0, 1, 2, 3, 4]
            f_idx = [0, 1, 2, 3, 4, 5, 6]
        else:
            raise ValueError("Insert a valid order, target couple of arguments")
    
    # PERMUTATIONS FOR CARTESIAN FUNCTIONS
    elif(basis == "cartesian"):
        opt1 = ("molden" in order.lower()) and ("pyscf" in target.lower())
        opt2 = ("pyscf" in order.lower()) and ("molden" in target.lower())
        opt3 = ("pyscf" in order.lower()) and ("horton" in target.lower())
        opt4 = ("pyscf" in order.lower()) and ("gamess" in target.lower())
        opt5 = ("gamess" in order.lower()) and ("pyscf" in target.lower())
        equal= (order.lower() == target.lower())
        
        if(opt1):
            p_idx = [1, 2, 0]
            d_idx = [0, 3, 4, 1, 5, 2]
            f_idx = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
        elif(opt2):
            p_idx = [2, 0, 1]
            d_idx = [0, 3, 5, 1, 2, 4]
            f_idx = [0, 6, 9, 3, 1, 2, 5, 8, 7, 4]
        elif(opt3):
            p_idx = [1, 2, 0]
            d_idx = [0, 1, 2, 3, 4, 5]
            f_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif(opt4):
            p_idx = [2, 0, 1]
            d_idx = [0, 3, 5, 1, 2, 4]
            f_idx = [0, 6, 9, 1, 2, 3, 7, 5, 8, 4]
        elif(opt5):
            p_idx = inverse_permutation([2, 0, 1])
            d_idx = inverse_permutation([0, 3, 5, 1, 2, 4])
            f_idx = inverse_permutation([0, 6, 9, 1, 2, 3, 7, 5, 8, 4])
        elif(equal):
            p_idx = [0, 1, 2]
            d_idx = [0, 1, 2, 3, 4, 5]
            f_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            raise ValueError("Insert a valid order, target couple of arguments")
    else:
        raise ValueError("The basis {} does not correspond to the possible basis set types (cartesian, spherical)".format(basis))

    # Define offsets for each shell
    p_off = len(p_idx)
    d_off = len(d_idx)
    f_off = len(f_idx)

    # Now perform the actual permutation
    for cnt, iarr in enumerate(bas):
        amom = iarr[1]
        if(amom == 0):
            idlist += [count]
            count += 1
        elif(amom == 1):
            temp    = np.arange(count, count + p_off)
            idlist += list(temp[p_idx])
            count += p_off
        elif(amom == 2):
            temp = np.arange(count, count + d_off)
            idlist += list(temp[d_idx])
            count += d_off
        elif(amom==3):
            temp = np.arange(count, count + f_off)
            idlist += list(temp[f_idx])
            count += f_off

    return(idlist)

def inverse_permutation(indices):
    """Return the inverse of a given permutation."""
    inverse = [0]*len(indices)
    for i in range(len(indices)):
        inverse[indices[i]] = i
    return(inverse)

def h5_to_pyscf(h5data):
    """Permute from h5 ordering to pyscf ordering.
       The bas input array contains the following structure:
       [[atom_number, n_of_basis_in_shell, max_amom, amom],
        [...],...]
       having length = number of basis functions
       The number n_of_basis_in_shell is the number of basis
       functions per given ang mom. shell, per atom.
       This is the structure found in the 'BASIS_FUNCTION_IDS' 
       key of a molcas hdf5 object, which can be used 
       as input. Ith returns the permuted indices
       as an 1D array.
    """
    h5data = np.array(h5data) # Transform the "BASIS_FUNCTION_IDS 'object' to a numpy 2D array"
    idlist = []
    # Add extra column with basis set indices to the
    # h5data array
    nbas   = len(h5data)
    natm   = np.max(h5data[:,0])
    extbas = np.hstack((np.arange(nbas).reshape((nbas, 1)), h5data))

    # Iterate over atoms
    for i in range(natm):
        # Reduced array for ith atom
        iat  = extbas[extbas[:, 1] == i + 1]
        # Max ang momenta per atom
        amoms = np.unique(iat[:,3])
        # Iterate over ang momenta per atom. These
        # are the shells.
        for j in amoms:
            # Count number of functions per shell (n_jbas).
            # Each shell contains all basis functions
            # of a given amom
            jbas   = iat[iat[:, 3] == j]
            n_jbas = np.max(jbas[:, 2])

            # This is the core of the algorithm:
            # 1) Reshape the basis indices of the given shell
            #    to (n_jbas, n_funct_per_shell)
            # 2) Transpose, then flatten
            part_id = jbas[:, 0].reshape((ang_mom_sp[j], n_jbas)).T.flatten()

            # At this point we have the molcas 1 ordering.
            # to get the pyscf ordering we have to 
            # permute the p orbitals: (+1,-1,0) ->(-1,0,+1)
#            if(j == 1):
#                part_id = part_id[n_jbas*[1, 2, 0]]
            if(j == 1):
                new_idx = []
                for k in range(n_jbas):
                    new_idx += [l + 3 * k for l in [1, 2, 0]]
                part_id = part_id[new_idx]
            idlist += part_id.tolist()
    return(idlist)

def pyscf_to_h5(bas):
    """Permute AOs from pyscf (internal) ordering
       to h5 (molcas hdf5, INPORB) ordering
    """
    idlist = []
    cn = 0
    natm = np.max(bas[:,0]) + 1
    print("natm = ", natm)
    # Iterate over atoms
    for i in range(natm):
        iat = bas[bas[:,0] == i]
        amoms = np.unique(iat[:,1])
        # Iterate over angular momenta
        for j in amoms:
            shls = len(iat[iat[:,1] == j])
            # Iterate over functions per shell.
            # Account for pyscf -> molcas permutation
            # (0, 1, 2) --> (2, 0, 1)
            if(j == 1):
                rng_amoms = [2, 0, 1]
            else:
                rng_amoms = range(ang_mom_sp[j])
#            for k in range(ang_mom_sp[j]):
            for k in rng_amoms:
                idlist += [cn + k + ang_mom_sp[j] * ishl for ishl in range(shls)]
            cn += ang_mom_sp[j] * shls
    return(idlist)

def permute_matrix(M, indices):
    """Permute matrix using indexing on the indices argument"""
    # Update:
    # transpose: permute cols first
    # Then permute rows
    # Finally transpose back so that the final matrix acts from the LEFT
#    return(np.transpose(M[indices])[indices])
    return((np.transpose(M[indices])[indices]).T)

#def perm_mixed_matrix(M, ind1, ind2):
#    """Permute a non-symmetric matrix using indexing on the indices argument"""
#    dim = len(M)
#    out_M = np.zeros((dim, dim))
#    for cnt, i in enumerate(range(dim)):
#        out_M[cnt] = M[ind1[cnt]][ind1]
#    return(out_M)

def perm_mixed_matrix(M, ind1, ind2):
    """Permute a non-symmetric matrix using indexing on the indices argument"""
    print("Shape before perm = ", M.shape)
    M_perm = M[ind1][:,ind2]
    print("Shape after perm = ", M_perm.shape)
    return(M_perm)
#    return(M[ind1][:,ind2])
#    dim1 = len(ind1)
#    dim2 = len(ind2)
#    out_M = np.zeros((dim1, dim2))
#    for cnt, i in enumerate(range(dim1)):
#        out_M[cnt] = M[ind1[cnt]][ind2]
#    return(out_M)
