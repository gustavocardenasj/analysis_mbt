from ctypes import cdll, POINTER, c_long, c_double
import numpy as np
import os
import platform
from copy import deepcopy
from molecule import Mol
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.misc import trans_cart2sph_norm as c2s_norm
from tools_mbt.misc import norm_cart as cnorm
from tools_mbt.misc import ang_mom_ct, ang_mom_sp, offcrt
from tools_mbt.misc import get_cart2sph, get_cartnorm
from tools_mbt.geom import Align, rotate_matrix

abspath = os.path.dirname(__file__) 
# Include intwrap.so location on the  LD_LIBRARY_PATH 

def symmetrize(M):
    """Symmetryze an upper triangular matrix"""
    return(M + np.transpose(M) - np.diagflat(M.diagonal()))

def wrap_ovlp_matrix():
    pform = platform.system()

    if(pform == "Linux"):
        dll  = cdll.LoadLibrary(abspath+"/intwrap.so")
    elif(pform == "Windows"):
        dll = cdll.LoadLibrary(abspath+"/intwrap.dll")
    else:
        raise OSError("Platform " + pform + " currently not supported")
    func = dll.intOvlpCrt
    func.argtypes = [POINTER(c_long), POINTER(c_long),
                     POINTER(c_double), POINTER(c_double),
                     c_long, c_long]

    return(func)

def wrap_ovlp_matrix_mix():
    """Compute overlap matrix between two sets of 
    different basis functions"""
    pform = platform.system()
    if(pform == "Linux"):
        dll = cdll.LoadLibrary(abspath+"/intwrap.so")
    elif(pform == "Windows"):
        dll = cdll.LoadLibrary(abspath+"/intwrap.dll")
    else:
        raise OSError("Platform " + pform + " currently not supported")
    func = dll.intOvlpCrtMix
    func.argtypes = [POINTER(c_long), POINTER(c_long), POINTER(c_double),
                     POINTER(c_long), POINTER(c_long), POINTER(c_double),
                     POINTER(c_double), c_long, c_long, c_long, c_long]
    return(func)

def calc_cart_ovlp_matrix(bas1, atm1, env1, ncrt1,
                     bas2 = None, atm2 = None, env2 = None, ncrt2 = None):
    """
    Wrapper for the computation of the cartesina unnormalized
    overlap matrix. All args are 1d flattened within the function
    """
    nshl1 = len(bas1)
    Ncrt1 = ncrt1
    Ncrt2 = ncrt1
    bas1  = bas1.flatten().ctypes.data_as(POINTER(c_long))
    atm1  = atm1.flatten().ctypes.data_as(POINTER(c_long))
    env1  = env1.flatten().ctypes.data_as(POINTER(c_double))
    print("nshl1 = ", nshl1, "Ncrt1 = ", ncrt1)
    if(ncrt2 == None):
        __wrap_ovlp_matrix = wrap_ovlp_matrix()
    else:
        nshl2 = len(bas2)
        Ncrt2 = ncrt2
        bas2  = bas2.flatten().ctypes.data_as(POINTER(c_long))
        atm2  = atm2.flatten().ctypes.data_as(POINTER(c_long))
        env2  = env2.flatten().ctypes.data_as(POINTER(c_double))
        print("nshl2 = ", nshl2, "Ncrt2 = ", ncrt2)
        __wrap_ovlp_matrix = wrap_ovlp_matrix_mix()


    # Initialize Overlap matrix
    S = np.zeros((Ncrt1, Ncrt2)).flatten().ctypes.data_as(POINTER(c_double))

    # Call function
    if(ncrt2 == None):
        __wrap_ovlp_matrix(bas1, atm1, env1, S, c_long(Ncrt1), c_long(nshl1))
    else:
        __wrap_ovlp_matrix(bas1, atm1, env1, 
                               bas2, atm2, env2,
                               S, c_long(Ncrt1), c_long(nshl1),
                               c_long(Ncrt2), c_long(nshl2))

    S = np.reshape(S[:Ncrt1**2], (-1,Ncrt1)) 
    return(S)

# Spherical Overlap Matrix. It takes as arguments two 
# Mol objects plus the cartesian overlap matrix.
def calc_sph_ovlp_matrix(Sxyz, mol1, mol2 = None):
    """
    Compute (normalized) spherical atomic overlap matrix 
    from the (unnormalized) cartesian overlap matrix
    """
    C1 = get_cart2sph(mol1)
    if(mol2 != None):
        C2 = get_cart2sph(mol2)
    else:
        C2 = C1

    Ssph = np.matmul(C1, np.matmul(Sxyz, np.transpose(C2)))
    return(Ssph)

# Normalized cartesian overlap matrix
def calc_cart_norm_ovlp_matrix(Sxyz, mol1, mol2 = None):
    """Compute normalized cartesian atomic overlap matrix from the (unnormalized)\
    cartesian overlap matrix
    """
    C1 = get_cartnorm(mol1)
    if(mol2 != None):
        C2 = get_cartnorm(mol2)
    else:
        C2 = C1

    Snorm = np.matmul(C1, np.matmul(Sxyz, np.transpose(C2)))
    return(Snorm)

# Functions requiring only Mol objects as input
def get_ao_overlap(mol1, mol2=None):
    """
    Compute AO overlap matrix for a Mol object, or
    between two mol objects. The same basis (cartesian or
    spherical) is assumed for both objects in the latter case. Return normalized matrix.

    IMPORTANT: if mol2 != None, it is assumed that mol1 and mol2 are aligned,
    so that any alignment must be performed outside of this function

    inplace_rotation: If True, performs a rotation of mol2 in-place, otherwise it creates a local copy on which to perform the rotation. Default = False
    """
    # Assess whether each mol comes from an orca_molden
    # file ( with primitives normalized)
    norm_1 = mol1.orca_molden

    # Get unnormalized cartesian overlap matrix
    if(mol2 != None):
        norm_2 = mol2.orca_molden
        # Consider QM atoms only in the alignment
#        qm_ids1 = np.unique(mol1._bas[:,0]).tolist() # QM atom indices
#        qm_ids2 = np.unique(mol2._bas[:,0]).tolist() # QM atom indices
        
        Scrt = calc_cart_ovlp_matrix(mol1._bas, 
                                     mol1._atm, 
                                     mol1._env, 
                                     mol1.ncrt,
                                     mol2._bas, 
                                     mol2._atm, 
                                     mol2._env, 
                                     mol2.ncrt)

        # Left and right normalization matrices
        
        C1_crt = get_cartnorm(mol1) 
        C2_crt = get_cartnorm(mol2) 
        C1_sph = get_cart2sph(mol1) 
        C2_sph = get_cart2sph(mol2) 
    else:
        Scrt = calc_cart_ovlp_matrix(mol1._bas, 
                                     mol1._atm, 
                                     mol1._env, 
                                     mol1.ncrt)
        # Left and right normalization matrices
        C1_crt = get_cartnorm(mol1) 
        C2_crt = C1_crt 
        C1_sph = get_cart2sph(mol1) 
        C2_sph = C1_sph

    # Then either normalize (if basis == cart) or
    # transform into (normalized) spherical overlap matrix
    if(mol1.basis == "spherical"):
        S = np.matmul(C1_sph, np.matmul(Scrt, C2_sph.T))
    elif(mol1.basis == "cartesian"):
        S = np.matmul(C1_crt, np.matmul(Scrt, C2_crt.T))
    else:
        raise ValueError(mol1.basis + " is not a valid basis set")

    return(S)

def get_mo_overlap(mol1, mol2 = None, Sao = None):
    """
    Compute MO overlap matrix for a Mol object, or
    between two mol objects. The same basis (cartesian or
    spherical) is assumed for both objects in the latter case.

    IMPORTANT: 

    if mol2 != None, mol1 and mol2 are assumed to be aligned, so that any
    alignment must be performed outside of this function

    inplace_rotation: If True, performs a rotation of mol2 in-place, otherwise it creates a local copy on which to perform the rotation. Default = False

    Sao = Atomic orbital overlap matrix, in case it has been computed prior
    """

    if(Sao != None):
        # Atomic overlap matrix provide as input
        Sao = Sao
    else:
        Sao = get_ao_overlap(mol1, mol2)

    if(mol2 != None):
        Smo = np.matmul(mol1.C_mo, np.matmul(Sao, mol2.C_mo.T))
    else:
        Smo = np.matmul(mol1.C_mo, np.matmul(Sao, mol1.C_mo.T))
    return(Smo)

if(__name__ == "__main__"):
    from argparse import ArgumentParser as ap

    parser = ap("Script to compute the (AO and MO) overlap matrix between two Mol objects")
    parser.set_defaults(f2 = None, basis = "spherical", rotation = 1)
    parser.add_argument("-f1", dest = "file1", type = str,\
                        help = "Input file for Mol 1")
    parser.add_argument("-f2", dest = "file2", type = str,\
                        help = "Input file for Mol 2. Default = None")
    parser.add_argument("-b", dest = "basis", type = str,\
                        help = "Basis type, either cartesian or spherical. Default = spherical")
    parser.add_argument("-r", dest = "rotation", type = int,\
                        help = "Perform an in-place alignment of the target coordinates. Default = 1. If 0 (False) create a deepcopy of the correspondin Mol object.")
    options       = parser.parse_args()
    file1         = options.file1
    file2         = options.file2
    basis         = options.basis
    rotation      = options.rotation
    center_origin = False
    
    # The targer geometry is rotated and displaced prior to performing
    # the overlaps calculations. This so as to avoid eventual undesired 
    # inplace rotations


    mol1 = Mol(file1, basis = basis, center_origin = center_origin)
    mol1.parse()
    if(options.file2 != None):
        mol2 = Mol(file2, basis = basis, center_origin = center_origin)
        mol2.parse()

        if(not rotation):
            # Create deepcopy of mol2
            tgt = deepcopy(mol2)
            tgt.align_molecule(mol1.atomcoords)

            # Align tgt to mol1, leaving mol2 unchanged

            Sao = get_ao_overlap(mol1, tgt)
            Smo = get_mo_overlap(mol1, tgt)
        else:
            # Rotate molecule in-place
            mol2.align_molecule(mol1.atomcoords)
            Sao = get_ao_overlap(mol1, mol2)
            Smo = get_mo_overlap(mol1, mol2)
    else:
        Sao = get_ao_overlap(mol1)
        Smo = get_mo_overlap(mol1)



