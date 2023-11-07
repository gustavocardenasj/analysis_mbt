#!/usr/bin/env python3
import numpy as np
from scipy.special import sph_harm, factorial, factorial2, lpmv
from tools_mbt.misc import *
from copy import deepcopy

################################################################################
############################## VARIOUS GRID TOOLS ##############################
################################################################################

class Cube_mol(object):

    def __init__(self, 
                 mol,
                 grid = [40, 40, 40],
                 data = "orbitals",
                 n_mo = 0,
                 box_width = 20):
        """
        Cube class that bears all the information necessary to 
        generate a gaussian cube file. Args:

        mol  = Mol object. Mandatory
        grid = 3D list with grid size on each dimension. Default: [40, 40, 40]
        data = data to be printed, default = orbitals
        n_mo = orbital to be printed
        box_width = width of the volumetric box
        """

        # The center is the COM of mol

        self.center = np.average(mol.atomcoords, axis = 0)        
        self.grid = grid
        self.data = data
        self.n_mo = n_mo
        self.box_w = box_width

        # Grid. 

        # Cartesian
        x = np.linspace(self.center[0] - box_width/2, self.center[0] + box_width/2, self.grid[0])
        y = np.linspace(self.center[1] - box_width/2, self.center[1] + box_width/2, self.grid[1])
        z = np.linspace(self.center[2] - box_width/2, self.center[2] + box_width/2, self.grid[2])

        # Width of each voxel:
        self.V  = [x[-1] - x[-2], y[-1] - y[-2], z[-1] - z[-2]]

        # 3D Meshgrid
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing = "ij")

        # Volumetric data 
        self.volume = None

    def generate_data(self, mol):
        """Generate volumetric data"""
        if(self.data == "orbitals"):
            self.volume = LCAO(mol, self.X, self.Y, self.Z, self.n_mo)
        elif(self.data == "density"):
            self.volume = electron_density(mol, self.X, self.Y, self.Z)
#        return(data)

def LCAO(mol, X, Y, Z, n_mo = 0):
    """
    Compute LCAO on a grid. Consider either cartesian
    or real spherical gaussians.

    Args: Mol object
    X, Y, Z = Cartesian volumetric grid
    n_mo = number of MO of interest

    """
    V = np.zeros(X.shape) # Volumetric data

    # Compute cartesian Gaussian functions
    # Iterate over basis shells
    cnt_ao  = 0 # Counter of AO
    prm_off = 0 # Offset for primitives
    for i, ibas in enumerate(mol._bas):
        amom     = ibas[1]
        if(mol.basis == "spherical"):
            n_contr = ang_mom_sp[amom]
            off_ang = off_angsph[amom]  # Offset for m (manetic q. number)
        elif(mol.basis == "cartesian"):
            n_contr  = ang_mom_ct[amom] # N of contractions per shell
            off_ang  = off_angcrt[amom] # Offset for [lx, ly, lz]
        else:
            raise(ValueError("Basis {} not supported".format(mol.basis)))
        center   = mol.atomcoords[ibas[0]] # Atomic coordinates

        # Accumulate primitive indices here. This is done to retrieve
        # pseudonormalization factors, that have to be removed
        prm_idx  = [prm_off + k for k in range(ibas[2])] 
        # Iterate over individual basis functions (contr.)
        for jbas in range(off_ang, off_ang + n_contr):
            lx, ly, lz = ANGCRT[jbas]
            amom_m     = ANGSPH[jbas]
            Cmo        = mol.C_mo[n_mo][cnt_ao]
            csum       = np.zeros(X.shape) # Accumulated contraction

            # Iterate over primitives, accumulate contraction
            for jpr in range(ibas[2]):
                exp_prm = mol._env[ibas[5] + jpr]
                cff_prm = mol._env[ibas[6] + jpr]
                inv_psd = 1/mol._pseudonorm[prm_idx[jpr]] # Remove pseudonormalization
                prefac  = inv_psd * cff_prm 
                csum += prefac * atomic_orbital(X, Y, Z, 
                                                amom, 
                                                amom_m, 
                                                lx, 
                                                ly, 
                                                lz, 
                                                exp_prm, 
                                                center, 
                                                basis = mol.basis, 
                                                normalize = True)

            # Accumulate LCAO
            V += Cmo * csum
            # Update AO counter
            cnt_ao    += 1
        # Update primitive offset
        prm_off += ibas[2]
    return(V)

def atomic_orbital(X, 
                   Y, 
                   Z, 
                   l = 0, 
                   m = 0, 
                   lx = 0, 
                   ly = 0, 
                   lz = 0, 
                   alpha = 0, 
                   center = [0., 0., 0.], 
                   basis = "spherical", 
                   normalize = False):
    """Generic function to compute a normalized AO on a grid"""
    if(normalize):
        N = gto_norm_const(l, lx, ly, lz, alpha, basis)
    else:
        N = 1.
    if(basis == "spherical"):
        return(N * atomic_orbital_sph_real(X, Y, Z, l, m, alpha, center))
    elif(basis == "cartesian"):
        return(N * atomic_orbital_xyz(X, Y, Z, lx, ly, lz, alpha, center))
    else:
        raise(ValueError("Basis type {} not supported".format(basis)))

def atomic_orbital_xyz(X, Y, Z, lx, ly, lz, alpha, center = [0., 0., 0.]):
    """
    Write an arbitrary cartesian Gaussian function computed on a grid.
    AOs are unnormalized.

    Args:

    X, Y, Z = Meshgrids on x, y, z
    lx, ly, lz = Angular momentum number for each coordinate
    alpha = exponent
    center = Coordinates of a given atomic center, default = [0, 0, 0]
     """
    # The calculation of N here makes the codeless efficient, but more compact


    W = (X-center[0])**lx * (Y-center[1])**ly * (Z-center[2])**lz * np.exp(-alpha * ((X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2))
    return(W)

def atomic_orbital_sph_real(X, Y, Z, l, m, alpha, center = [0., 0., 0.]):
    """
    Write an arbitrary real spherical Gaussian function computed on a grid. 
    Coordinates are spherical.

    X = Radial coordinate
    P = phi - azimuth (in [0, 2*pi]) In the exp.
    T = theta - polar in [0, pi]  In the Legendre polynomial
    

    The returned function is unnormalized
    """
    # The calculation of N here makes the codeless efficient, but more compact
    # Transform coordinates from cartesian to 
    # Spherical
    R = ((X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)**0.5
    P = np.arctan2(Y-center[1], X-center[0]) # Azimuth [0, 2*pi]
    T = np.arctan2(((X-center[0])**2 + (Y-center[1])**2)**0.5, Z-center[2]) # Colatitudinal [0, pi] 
    # Unnormalized Radial part
    W = R**l * np.exp(-alpha * R**2) 
    # Spherical harmonic LC. m-entry with the absolute value. 
    # The CS factor is included EXPLICITLY.
    # Schlegel norm. Factor - we are NOT using regulas solid SH (no factor 4pi/(2l+1))

    # USING REAL ALGEBRA (Schlegel Norm with factor 2)

    N = (((2*l + 1)/(4*np.pi)) * factorial(l - abs(m))/factorial(l + abs(m)))**0.5 
    if(m>0):
        S = (-1)**m * N * lpmv(abs(m), l, np.cos(T)) * np.cos(abs(m)*P)/2**0.5
    elif(m == 0):
        S = N * lpmv(abs(m), l, np.cos(T))
    else:
        S = (-1)**m * N * lpmv(abs(m), l, np.cos(T)) * np.sin(abs(m)*P)/2**0.5

#    # USING COMPLEX ALGEBRA (Schlegel Norm with factor 2)
#    Y = sph_harm(np.abs(m),l,P,T)
#    if(m>0):
#        S = (-1.0)**m * 2**0.5 * Y.real
#    elif(m == 0):
#        S = N * Y
#    else:
#        S = (-1.0)**m * (2)**0.5 * Y.imag
#
    return((W * S).astype("float64"))

def electron_density(mol, X, Y, Z):
    """
    Calculate the electron density of a molecule on a grid
    from its  molecular orbitals. Assume natural orbitals and the
    corresponding occupations. Formula:
    rho(r) = sum_i n_i * psi_i(r) * conj(psi_i(r))
    psi_i = ith natural orbital
    n_i   = occupation of psi_i(r)
    """
    if(not(hasattr(mol, "nele"))):
        raise(ValueError("Electron density calculation cannot be performed without knowing the number of electrons!"))

    # Define density grid
    V = np.zeros(X.shape)
    # Iterate over MOs. Support for closed shells only
    occ_mos = np.where(mol.mooccnos > 1e-3)[0]
    for imo in occ_mos:
        V += mol.mooccnos[imo] * LCAO(mol, X, Y, Z, n_mo = imo)**2
    print("Max V = ", np.max(V))
    return(V)

