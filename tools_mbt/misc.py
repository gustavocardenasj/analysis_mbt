#!/usr/bin/env python3
from math import pi
import numpy as np
from scipy.special import factorial, factorial2

def gto_sph_norm(n, a):
    """Prefactor to pseudo-normalize cartesian gtos.
       Missing prefactor (4*pi/(2*n + 1))**0.5.
       The actual multiplication occurs when we transform
       Sxyz into Ssph
    """
    norm = (2**(2*n + 3) * factorial(n+1) * (2*a)**(n + 1.5)\
            /(factorial(2*n + 2) * pi**0.5) )**0.5
    return(norm)

def gto_cart_norm(l, a):
    """
    Prefactor to pseudo-normalize cartesian gtos.
    In the case of a cartesian gaussian basis, the 
    missing factors will be tabulated below.
    """
    norm = (2**(2*l) * a**(l + 1.5)/pi**(1.5))**0.5
    return(norm)

#def gto_cart_norm_abs(lx, ly, lz, a):
#    """
#    Compute normalization factor of an unnormalized
#    cartesian function
#    """
#    l = lx + ly + lz
#    norm = gto_cart_norm(l ,a) * (factorial(lx)*factorial(ly)*factorial(lz)/factorial(2*lx)*factorial(2*ly)*factorial(2*lz))**0.5
#    return(norm)

def gto_norm_const(l = 0, 
                   lx = 0, 
                   ly = 0, 
                   lz = 0,
                   alpha = 0,
                   basis = "spherical"):
    """
    Compute normalization constant for a given Gaussian
    primitive. Supports cartesian and pure spherical Gaussians
    """
    if(basis == "spherical"):
        return(gto_sph_norm_abs(l, alpha))
    elif(basis == "cartesian"):
        return(gto_cart_norm_abs(lx, ly, lz, alpha))
    else:
        raise(ValueError("Basis type {} not supported".format(basis)))
    

def gto_cart_norm_abs(lx, ly, lz, alpha):
    """
    Compute normalization factor of an unnormalized
    cartesian function
    """
    l = lx + ly + lz
#    # Schlegel formula IJQC, 54, 83-87 (1995)
#    return(((2**(2*l) * factorial(lx) * factorial(ly) * factorial(lz) * alpha**( l + 1.5))/(factorial(2*lx) * factorial(2*ly) * factorial(2*lz)*pi**(1.5)))**0.5)
    
    # HORTON
    return((((2*alpha/pi)**1.5 * (4*alpha)**l)/(factorial2(2*lx - 1)*factorial2(2*ly - 1) * factorial2(2*lz - 1)))**0.5)

def gto_sph_norm_abs(n, alpha):
    """
    Compute normalization factor of an unnormalized
    real spherical function. Formula from HORTON
    """
    # Horton (takes into account the prefactor on the radial part)
#    return(((2*alpha/pi)**1.5 * (4*alpha)**n/factorial2(2*n - 1))**0.5)

    # Schlegel formula on PySCF
    return(((2**(2*n + 3) * factorial(n + 1) * (2*alpha)**(n + 1.5))/(factorial(2*n + 2) * pi**0.5))**0.5)

#    # Schlegel formula IJQC, 54, 83-87 (1995)
#    return(((2**(2*n + 3) * factorial(n + 1) * alpha**(n + 1.5))/(factorial(2*n + 2) * pi**0.5))**0.5)

def gto_sph_norm_orca(n, alpha):
    """
    Compute normalization factor used in Orca 
    Here the 1/factorial2(2*n - 1)**0.5 is missing
    """
    return(((2*alpha/pi)**1.5 * (4*alpha)**n)**0.5)

def get_cart2sph(mol, normalized = False):
    """
    Build (un)normalized cartesian to (normalized) spherical 
    basis function transformation matrix
    """
    if(normalized):
        c2s_ref = trans_cart2sph_norm
    else:
        c2s_ref = trans_cart2sph
    bas = mol._bas
    nct = mol.ncrt
    nsp = mol.nsph
    C   = np.zeros((nsp, nct))

    # row, col counters of C
    i_row = 0
    j_col = 0
    for cn1, ishl in enumerate(bas):
        # Number of spherical/cartesian functions per shell
        isp = ang_mom_sp[ishl[1]]
        ict = ang_mom_ct[ishl[1]]
        # Iterate over rows
        for cn2 in range(isp):
            # Ierate over cols
            for cn3 in range(ict):
                offset = offcrt[ishl[1]]
                c2s_idx = offset + cn3 + cn2*ict
                C[i_row + cn2][j_col + cn3] = c2s_ref[c2s_idx]
        j_col += ict
        i_row += isp
    return(C)

def get_cartnorm(mol):
    """
    Build cartesian basis function normalization matrix
    """
    bas = mol._bas
    nct = mol.ncrt
    C   = np.zeros((nct, nct))

    # row, col counters of C
    i_row = 0
    j_col = 0
    # Iterate over rows
    for cn1, ishl in enumerate(bas):
        # Number of cartesian functions per shell
        ict = ang_mom_ct[ishl[1]]
        # Iterate over rows
        for cn2 in range(ict):
            # iterate over cols
            for cn3 in range(ict):
                offset = offcrt[ishl[1]]
                cnorm_idx = offset + cn3 + cn2*ict
                C[i_row + cn2][j_col + cn3] = norm_cart[cnorm_idx]
        j_col += ict
        i_row += ict
    return(C)

################## CONVERSION UNITS ##################

# Conversion units
#Distance
ag2bohr = 1.8897259886 
bohr2ag = 0.529177249

# Energy
kc2ev = 0.043364115309
kc2ha = 0.0015936010974
kc2cm_1 = 349.75501122
kc2j = 4.1840000

ha2ev = 27.211399
ha2kc = 627.50960803
ha2cm_1 = 219474.63

ev2ha = 0.036749304951
ev2kc = 23.060541945
ev2cm_1 = 8065.5401069

# Other
# Ha/Angs^2 --> millidine/Angs
ha2mdine_a = 4.3597442793

# Ha/bohr^2 --> millidine/Angs
ha2mdine_b = 15.569296023

# Assess if a string is float
def is_float(s):
    try:
        float(s)
        return(True)
    except(ValueError):
        return(False)
####################### CONSTANTS ####################

# Number of contractions per shell
ang_mom_ct = [1, 3, 6, 10]
ang_mom_sp = [1, 3, 5, 7]

# Angular momentum numbers
ang_mom = {"s": 0,
           "p": 1,
           "d": 2,
           "f": 3,
           "g": 4}

# Atom numbers <--> Atom names
atom_nums  = {"H": 1, "He": 2, "Li": 3, "Be": 4, "Al": 5, "C": 6,\
        "N": 7, "O": 8, "F": 9, "Ne": 10, "P": 15, "S": 16, "Cl": 17,\
        "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,\
        "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni":28, "Cu": 29,\
        "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se":34, "Br": 35,\
        "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41,\
        "Mo":42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47,\
        "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,\
        "Xe": 54, "Cs": 55, "Ba": 56, "Pt": 78, "Au": 79 }

atom_names = {atom_nums[key]: key for key in atom_nums.keys()}

# Atomic weights. {0:0} corresponds to the ghost atom situation
weights    = {0: 0.0000, 1: 1.0008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81,\
              6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,\
              11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974,\
              16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098, 20: 40.078,\
              21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938,\
              26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38,\
              31: 69.723, 32: 72.630, 33: 74.922, 34: 78.971, 35: 79.904,\
              36: 83.978, 37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224,\
              41: 92.906, 42: 95.95, 43: 106.42, 44: 101.07, 45: 102.91,\
              46: 106.42, 47: 107.87, 48: 112.41, 49: 114.82, 50: 118.71,\
              51: 121.76, 52: 127.60, 53: 126.90, 54: 131.29, 55: 132.91,\
              56: 137.33, 78: 195.08, 79: 196.97}

# Offsets for cartesian  and spherical functions
offcrt = [0, 1, 10, 40]
off_angcrt = [0, 1, 4, 10]
off_angsph = [0, 1, 4, 9]

ANGSPH = [0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2]

ANGCRT = [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [2, 0, 0],
          [1, 1, 0],
          [1, 0, 1],
          [0, 2, 0],
          [0, 1, 1],
          [0, 0, 2],
          [3, 0, 0],
          [2, 1, 0],
          [2, 0, 1],
          [1, 2, 0],
          [1, 1, 1],
          [1, 0, 2],
          [0, 3, 0],
          [0, 2, 1],
          [0, 1, 2],
          [0, 0, 3]]
# PYSCF ordering:
# P: Y, Z, X
# D: DX2, DXY, DXZ, DY2, DYZ, DZ2
# F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
#    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9

# THESE FACTORS NORMALIZE THE CARTESIAN FUNCTIONS,
# SO THAT THE SPHERICAL ORBITALS BE NORMALIZED AS WELL
trans_cart2sph = \
        [
        #S
        (1./(4*pi))**0.5,
        #PY
        (3./(4*pi))**0.5,
        0.,
        0.,
        #PZ
        0.,
        (3./(4*pi))**0.5,
        0.,
        #PX
        0.,
        0.,
        (3./(4*pi))**0.5,
        #DXY  (-2)
        0.,
        1.092548430592079070,
        0.,
        0.,
        0.,
        0.,
        #DYZ  (-1)
        0.,
        0.,
        0.,
        0.,
        1.092548430592079070,
        0.,
        #DZ2  (0)
        -0.315391565252520002,#*1.4142135623730951,  # extra sqrt(2),
        0.,
        0.,
        -0.315391565252520002,#*1.4142135623730951,  # extra sqrt(2),
        0.,
        0.630783130505040012,#*1.4142135623730951,  # extra sqrt(2),
        #DXZ  (+1)
        0.,
        0.,
        1.092548430592079070,
        0.,  
        0., 
        0.,
        #DY2  (+2)
#        0.546274215296039535*1.4142135623730951,  # extra sqrt(2),
#        1.092548430592079070, 
        0.546274215296039535,
#        1.092548430592079070/(2.**0.5), 
        0.,
        0.,
#        -1.092548430592079070,
#        -1.092548430592079070/(2.**0.5),
        -0.546274215296039535,
#        -0.546274215296039535*1.4142135623730951,  # extra sqrt(2),
        0.,
        0.,
        #F-3 Y(3X2 - Y2)  S(3,3)
        0.,
        1.770130769779930531,
        0.,
        0.,
        0.,
        0.,
        -0.590043589926643510,
        0.,
        0.,
        0.,
        #F-2 XYZ S(3,2)
        0.,
        0.,
        0.,
        0.,
        2.890611442640554055,
        0.,
        0.,
        0.,
        0.,
        0.,
        #F-1 YZ2 S(3,1)
        0.,
        -0.457045799464465739,
        0.,
        0.,
        0.,
        0.,
        -0.457045799464465739,
        0.,
        1.828183197857862944,
        0.,
        #F0 Z3 C(3,0)
        0.,
        0.,
        -1.119528997770346170,
        0.,
        0.,
        0.,
        0.,
        -1.119528997770346170,
        0.,
        0.746352665180230782,
        #F+1 XZ2 C(3,1)
        -0.457045799464465739,
        0.,
        0.,
        -0.457045799464465739,
        0.,
        1.828183197857862944,
        0.,
        0.,
        0.,
        0.,
        #F+2 Z(X2 - Y2) C(3,2)
        0.,
        0.,
        1.445305721320277020,
        0.,
        0.,
        0.,
        0.,
        -1.445305721320277020,
        0.,
        0.,
        #F+3 X(X2- 3Y2) C(3,3)
        0.590043589926643510,
        0.,
        0.,
        -1.770130769779930530,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        ]

# Same as before, but with
# normalization being taken into account
# PYSCF ordering:
# P: Y, Z, X
# D: DX2, DXY, DXZ, DY2, DYZ, DZ2
# F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
#    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9
trans_cart2sph_norm = [
        # S
        1.,
        # PY
        0.,
        1.,
        0.,
        # PZ
        0.,
        0.,
        1.,
        # PX
        1.,
        0.,
        0.,
        # DXY (-2)
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        # DYZ (-1)
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        # DZ2 (0)
        -0.2886751345948129,#-0.5, # 1/(2*sqrt(3))
        0.,
        0.,
        -0.2886751345948129,#-0.5, 1/(2*sqrt(3))
        0.,
        0.5773502691896258, # 1., 1/(sqrt(3))
        # DXZ (+1)
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        # DY2 (+2)
        0.5, #0.8660254037844386, # sqrt(3)/2
        0.,
        0.,
        -0.5, #-0.8660254037844386, # -sqrt(3)/2
        0.,
        0.,
        # F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
        #    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9
        #F-3 Y(3X2 - Y2)  S(3,3)
        0.,
        1.0606601717798214, # 3*sqrt(4)/2
        0.,
        0.,
        0.,
        0.,
        -0.7905694150420949, # -sqrt(10)/4
        0.,
        0.,
        0.,
        #F-2 XYZ S(3,2)
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        0.,
        #F-1 YZ2 S(3,1)
        0.,
        -0.27386127875258304, # -sqrt(30)/20
        0.,
        0.,
        0.,
        0.,
        -0.6123724356957945, # -sqrt(6)/4 
        0.,
        1.0954451150103321, # sqrt(30)/5
        0.,
        #F0 Z3 C(3,0)
        0.,
        0.,
        -0.223606797749979, # -sqrt(3)/5
        0.,
        0.,
        0.,
        0.,
        -0.223606797749979, # -sqrt(3)/5
        0.,
        1.,
        #F+1 XZ2 C(3,1)
        -0.6123724356957945, # -sqrt(6)/4
        0.,
        0.,
        -0.27386127875258304, # -sqrt(30)/20
        0.,
        1.0954451150103321, # sqrt(30)/5
        0.,
        0.,
        0.,
        0.,
        #F+2 Z(X2 - Y2) C(3,2)
        0.,
        0.,
        0.8660254037844386, # sqrt(3)/2
        0.,
        0.,
        0.,
        0.,
        -0.8660254037844386, # -sqrt(3)/2
        0.,
        0.,
        #F+3 X(X2- 3Y2) C(3,3)
        0.7905694150420949, # sqrt(10)/4
        0.,
        0.,
        -0.8660254037844386, # -sqrt(3)/2
        0.,
        0.,
        0.,
        0.,
        0.,
        0.
        ]

norm_cart = \
        [
        # S
        (1./(4*pi))**0.5,
        # PY
        0.,
        (3./(4*pi))**0.5,
        0.,
        # PZ
        0.,
        0.,
        (3./(4*pi))**0.5,
        # PX
        (3./(4*pi))**0.5,
        0.,
        0.,
        # DX2
        0.630783130505040012,
        0.,
        0.,
        0.,
        0.,
        0.,
        # DXY
        0.,
        1.092548430592079070,
        0.,
        0.,
        0.,
        0.,
        # DXZ
        0.,
        0.,
        1.092548430592079070,
        0.,
        0.,
        0.,
        # DY2
        0.,
        0.,
        0.,
        0.630783130505040012,
        0.,
        0.,
        # DYZ
        0.,
        0.,
        0.,
        0.,
        1.092548430592079070,
        0.,
        # DZ2
        0.,
        0.,
        0.,
        0.,
        0.,
        0.630783130505040012,
        # FX3
        0.7463526651802308,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        # FXXY
        0.,
        1.6688952945311362,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        # FXXZ
        0.,
        0.,
        1.6688952945311362,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        # FXYY
        0.,
        0.,
        0.,
        1.6688952945311362,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        # FXYZ
        0.,
        0.,
        0.,
        0.,
        2.890611442640554055,
        0.,
        0.,
        0.,
        0.,
        0.,
        # FXZZ
        0.,
        0.,
        0.,
        0.,
        0.,
        1.6688952945311362,
        0.,
        0.,
        0.,
        0.,
        # FY3
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.7463526651802308,
        0.,
        0.,
        0.,
        # FYYZ
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        1.6688952945311362,
        0.,
        0.,
        # FYZZ
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        1.6688952945311362,
        0.,
        # FZ3
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.7463526651802308
        ]

# From Schlegel, 1995
# Cartesians to Pure (complex) spherical Gaussians.
# Normalization of both cartesians and pure Gaussians
# IS accounted for.
#
# The ordering of the cartesian functions is the usual
# PySCF one. The ordering of the Pure Gaussians
# has increasing m quantum number (-l, .., 0, .., +l)
# PYSCF ordering:
# P: Y, Z, X
# D: DX2, DXY, DXZ, DY2, DYZ, DZ2
# F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
#    0  , 1   , 2   , 3   , 4   , 5   , 6  , 7   , 8   , 9

trans_cart2pure = [
        # S
        1,
        # PY (-1)
        -1j*0.7071067811865475,
        0,
        0.7071067811865475,
        # PZ (0)
        0,
        1,
        0,
        # PX (+1)
        1j*0.7071067811865475,
        0,
        0.7071067811865475,
        # D(-2)
        0.6123724356957945,
        -1j*0.7071067811865475,
        0,
        -0.6123724356957945,
        0,
        0,
        # D(-1)
        0,
        0,
        0.7071067811865475,
        0,
        -1j*0.7071067811865475,
        0,
        # D(0)
        -0.5,
        0,
        0,
        -0.5,
        0,
        1,
        # D(+1)
        0,
        0,
        0.7071067811865475,
        0,
        1j*0.7071067811865475,
        0,
        # D(+2)
        0.6123724356957945,
        1j*0.7071067811865475,
        0,
        -0.6123724356957945,
        0,
        0,
        # Redundant S function: (x2 + y2 + z2)
        1,
        0,
        0,
        1,
        0,
        1,
        # F: FX3, FXXY, FXXZ, FXYY, FXYZ, FXZZ, FY3, FYYZ, FYZZ, FZ3
        # F: 300, 210,  201,  120,  111,  102,  030, 021,  012,  003
        # F(-3)
        0.5590169943749475,
        -0.75*1j,
        0,
        -0.75,
        0,
        0,
        1j*0.5590169943749475,
        0,
        0,
        0,
        # F(-2)
        0,
        0,
        0.6123724356957945,
        0,
        -1j*0.7071067811865475,
        0,
        0,
        -0.6123724356957945,
        0,
        0,
        # F(-1)
        -0.4330127018922193,
        1j*0.19364916731037085,
        0,
        -0.19364916731037085,
        0,
        0.7745966692414834,
        1j*0.4330127018922193,
        0,
        -1j*0.7745966692414834,
        0,
        # F(0)
        # F: 300, 210,  201,  120,  111,  102,  030, 021,  012,  003
        0,
        0,
        -0.6708203932499369,
        0,
        0,
        0,
        0,
        -0.6708203932499369,
        0,
        1,
        # F(+1)
        -0.4330127018922193,
        -1j*0.19364916731037085,
        0,
        -0.19364916731037085,
        0,
        0.7745966692414834,
        -1j*0.4330127018922193,
        0,
        1j*0.7745966692414834,
        0,
        # F(+2)
        0,
        0,
        0.6123724356957945,
        0,
        1j*0.7071067811865475,
        0,
        0,
        -0.6123724356957945,
        0,
        0,
        # F(-3)
        0.5590169943749475,
        0.75*1j,
        0,
        -0.75,
        0,
        0,
        -1j*0.5590169943749475,
        0,
        0,
        0,
        # F: 300, 210,  201,  120,  111,  102,  030, 021,  012,  003
        # Redundant P functions:
        # XR2
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        # YR2
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        # ZR2
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1
        ]

