#########################################################
# MODULE TO HANDLE AO ROTATIONS            #
#########################################################

import numpy as np
from scipy.special import factorial # Implement or tabulate
from scipy.linalg import block_diag
from tools_mbt.misc import atom_nums, ag2bohr,\
        ang_mom, ang_mom_ct, ang_mom_sp,\
        gto_sph_norm, trans_cart2pure,\
        offcrt
#from molden_writer import permute_matrix
from tools_mbt.permutations import permute_matrix

def sh_cart2cplx(l):
    """Get the transformation matrix from real
       cartesian Gaussians to pure
       spherical harmonics
    """
    # Matrix includes redundant s/p... functions
    Ncrt  = ang_mom_ct[l]
    T = np.zeros((Ncrt, Ncrt), dtype = complex)
    offset = offcrt[l]
    for irow in range(Ncrt):
        for jcol in range(Ncrt):
            idx = offset + irow*Ncrt + jcol
            T[irow][jcol] = trans_cart2pure[idx]
    return(T)

def sh_cplx2cart(l):
    """Get the transformation matrix from pure to real
       Cartesian Gaussians
       l = angular momentum
    """
    T = sh_cart2cplx(l)
    # Calculate Pseudo-inverse
    return(np.linalg.pinv(T))

def sh_real2cplx(l):
    """Get the transformation matrix from real to pure
       spherical harmonics.

       N.B.: This matrix is the complex conjugate of that
       on the 1997 Theochem paper!
    """
    U = np.zeros((2*l + 1, 2*l + 1), dtype = complex)
    realfc = 0.5**0.5
    imfc   = 1j*realfc
    
    # Coefficient for the "central spherical harmonic"
    U[l][l] = 1
    #Matrix shape: (realfc) *
    #[[-i, 0, ...,   0,     i(-1)**l],
    # [0, -i, ...,  i(-1)**(l-1),  0],
    # [0, ..., ...,    ...,        0],
    # [0,..., 1, ...,              0],
    #[[0, ..., ..., ..., ...,      0],
    # [0, 1, ...,   (-1)**(l-1),   0],
    # [1, 0, ..., .., 0,     (-1)**l]
    for ml in range(1, l+1):
        U[l-ml][l-ml] = -imfc
        U[l+ml][l-ml] = realfc
        if(ml%2 == 0):
            U[l-ml][l+ml] = imfc
            U[l+ml][l+ml] = realfc
        else:
            U[l-ml][l+ml] = -imfc
            U[l+ml][l+ml] = -realfc
    return(U)

def sh_cplx2real(l):
    """Get the transformation matrix from pure to real
       spherical harmonics
       l = angular momentum
    """
    U = sh_real2cplx(l)
    return(np.transpose(U.conj()))

def wigner_D_real(l, D):
    """Transform Wigner D rotation matrix from pure to
       real.
       l = angular momentum
       D = Wigner D matrix in the pure spherical harmonic basis
    """
    # Formula:
    # UT * d * U
    # where U is the unitary transformation matrix
    # from pure to real spherical harmonics

    U  = sh_cplx2real(l)
    UT = np.transpose(U.conj())
    Dreal = np.einsum("ij,jk,kl->il", UT, D, U).real
    return(Dreal)

def wigner_D(l, alpha, beta, gamma, real = True):
    """Get wigner D rotation matrix
       l = angular momentum
       alpha, beta, gamma = Euler angles in the Z-Y-Z convention
       real = bool (Whether to get it on the real or pure spherical 
       harmonic basis)
    """
    #Dm,m' = <lm|R(alpha,beta,gamma|lm')
    d = wigner_d_small(l, beta)
    ml = np.arange(-l, l+1)
    D = np.einsum("i,ij,j->ij", np.exp(-1j*alpha*ml),
                                d,
                                np.exp(-1j*gamma*ml))
    if(real):
        D = wigner_D_real(l, D)
    return(D)

def wigner_d_small(l, beta):
    """Get Wigner small rotation matrix d_mm'^l(beta).
       Convention Z-Y-Z is assumed. 
       Returns: d small matrix (rotation of pure-complex
                spherical harmonics)
    """
    bh = beta/2
    c  = np.cos(beta) 
    s  = np.sin(beta) 
    ch = np.cos(bh)
    sh = np.sin(bh)
    s2b = np.sin(2*beta)
    if(l == 0):
        dsmall = np.array([[1.]])
    elif(l == 1):
        # Molcas ordering: -1, 0, 1 (y, z, x)
        dsmall = np.array([[ch**2, (2)**-0.5*s, sh**2],
                           [-(2)**-0.5*s, c, 2**-0.5*s],
                           [sh**2, -(2)**-0.5*s, ch**2]])
    elif(l == 2):
        # From 1994 NMR notes (Rotations, Euler Angles and
        # Wigner Rotation Matrices) but with ordering 
        # [-2, -1, 0, 1, 2]
        c2  = c**2
        s2  = s**2
        ch2 = ch**2
        sh2 = sh**2
        ch4 = ch2**2
        sh4 = sh2**2
        d0  = (3./8.)**0.5 * s2
        s2bf = np.sqrt(3./8.)*2*s*c
        ct   = (3*c2 - 1)/2
        c2sh2 = c2 - sh2
        ch2c2 = ch2 - c2
        dsmall = np.array([[ch4, ch2*s, d0 , sh2*s, sh4],

                           [-ch2*s , c2sh2,  s2bf, ch2c2, sh2*s],

                           [d0, -s2bf, ct , s2bf, d0],

                           [-sh2*s,  ch2c2, -s2bf, c2sh2, ch2*s ],

                           [sh4, -sh2*s ,d0 ,-ch2*s , ch4]])
    else:
        # J. J. Sakurai p. 223
        amoms = np.arange(2*l + 1)
        fct = factorial(amoms)
        cs  = ch**(amoms)
        ss  = sh**(amoms)
        dsmall = np.zeros((2*l + 1, 2*l + 1))
        for cn1, m1 in enumerate(range(-l, l+1)):
            for cn2, m2 in enumerate(range(-l, l+1)):
                num   = np.sqrt(fct[l+m2] * fct[l-m2]\
                               *fct[l+m1] * fct[l-m1])
                for k in range(max(m2-m1,0), min(l+m2, l-m1)+1):
#                for k in range(min(m2-m1,0), max(l+m2, l-m1)+1):
                    denom = (fct[l+m2-k] * fct[l-m1-k]\
                            *fct[m1-m2+k] * fct[k])
                    dsmall[cn1][cn2] += (-1)**(k-m2+m1) \
                            *  cs[2*l-2*k+m2-m1] * ss[2*k-m2+m1]\
                            / denom
                dsmall[cn1][cn2] *= num
    return(dsmall)

def cartesian_D(l, alpha, beta, gamma):
    """
    Get rotation matrix for cartesian orbitals

    Args:

    l = Angular momentum

    alpha, beta, gamma = Euler angles
    """

    # Factors:

    # Normal 
    cg  = np.cos(gamma)
    cb  = np.cos(beta)
    ca  = np.cos(alpha)
    sg  = np.sin(gamma)
    sb  = np.sin(beta)
    sa  = np.sin(alpha)
    # Squares
    cg2 = cg**2
    cb2 = cb**2
    ca2 = ca**2
    sg2 = sg**2
    sb2 = sb**2
    sa2 = sa**2
    # Double angles
    c2g = np.cos(2*gamma)
    c2b = np.cos(2*beta) 
    c2a = np.cos(2*alpha)
    s2g = np.sin(2*gamma)
    s2b = np.sin(2*beta) 
    s2a = np.sin(2*alpha)

    if(l == 0):
        d_crt = np.array([[1.]])
    elif(l == 1):
        # Matrix corresponds to standard
        # rotation in R3. Return on [1, 2, 0]
        # ordering
#        d_crt = permute_matrix(euler_to_rot(alpha, beta, gamma, from_right = False),[1,2,0])
#        d_crt = permute_matrix(euler_to_rot(alpha, beta, gamma),[1,2,0])
        d_crt = permute_matrix(euler_to_rot(alpha, beta, gamma),[1,2,0]) # Update: from_right = False updated.
                                                                         # Transposition exclusively due to permutation
    elif(l == 2):
        # Computed by hand
        d_crt = [# X2
                 [(cg*cb*ca - sg*sa)**2,
                  s2g*(sa2 - cb2*ca2) - c2g*cb*s2a,
                  cg*s2b*ca2 - sg*sb*s2a,
                  (sg*cb*ca + cg*sa)**2,
                  -(sg*s2b*ca2 + cg*sb*s2a),
                  sb2*ca2],
                 # XY
                 [0.5*(s2a*(cg2*cb2 - sg2) + s2g*cb*c2a),
                  c2g*cb*c2a - 0.5*s2g*(1 + cb2)*s2a,
                  0.5*cg*s2b*s2a + sg*sb*c2a,
                  0.5*(s2a*(sg2*cb2 - cg2) - s2g*cb*c2a),
                  cg*sb*c2a - 0.5*sg*s2b*s2a,
                  0.5*sb2*s2a],
                 # XZ
                 [0.5*(s2g*sb*sa - cg2*s2b*ca),
                  0.5*s2g*s2b*ca + c2g*sb*sa,
                  cg*c2b*ca - sg*cb*sa,
                  -0.5*(sg2*s2b*ca + s2g*sb*sa),
                  -sg*c2b*ca - cg*cb*sa,
                  sb*cb*ca],
                 # Y2
                 [(cg*cb*sa + sg*ca)**2,
                  s2g*(ca2 - cb2*sa2) + c2g*cb*s2a,
                  cg*s2b*sa2 + sg*sb*s2a,
                  (sg*cb*sa - cg*ca)**2,
                  cg*sb*s2a - sg*s2b*sa2,
                  sb2*sa2
                 ],
                 # YZ
                 [-0.5*(cg2*s2b*sa + s2g*sb*ca),
                  -c2g*sb*ca + 0.5*s2g*s2b*sa,
                  cg*c2b*sa + sg*cb*ca,
                  0.5*(-sg2*s2b*sa + s2g*sb*ca),
                  cg*cb*ca - sg*c2b*sa,
                  sb*cb*sa
                 ],
                 # Z2
                 [cg2*sb2,
                  -s2g*sb2,
                  -cg*s2b,
                  sg2*sb2,
                  sg*s2b,
                  cb2
                 ]
                ]
#        d_crt = np.transpose(d_crt) # Update: Make it act from left
    else:
        raise ValueError("Rotations for cartesian orbitals with L>2 are not supported")
#    return(np.transpose(d_crt))
    return(d_crt)

def euler_to_rot(alpha, beta, gamma, from_right=False):
    """
    Get the rotation matrix in the Z-Y-Z convention from
    a given triad of euler angles
    from_right = Whether the resulting R matrix acts from the 
                 right on a NX3 geometry. e.g.: M * R. 
                 Default = True

    The returned matrix has the ordering [x,y,z], i.e:
    
    [0,1,2]

    but recall that the internal ordering for Mol objects is 

    [1,2,0]

    """
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

    R = [[ca*cb*cg - sa*sg, -cg*sa - ca*cb*sg, ca*sb],
         [ca*sg + cb*cg*sa, ca*cg - cb*sa*sg, sa*sb],
         [-cg*sb, sb*sg, cb]]
    if(from_right):
        R = np.transpose(R)
    return(np.array(R))

def rot_to_euler(R, from_right = False):
    """Get euler angles in the Z-Y-Z convention from a
       given rotation matrix
       from_right = Whether the input R matrix acts from the 
                    right on a NX3 geometry. e.g.: M * R. 
                    Default = True
    """
    if(not from_right):
        R = np.transpose(R)
    # alpha
    if(R[0][2] == 0):
        alpha = np.pi/2
    else:
        alpha = np.arctan2(R[1][2], R[0][2])

    # beta
    if(R[2][2]**2 > 1):
        raise ValueError("R[2][2] cannot be > 1 or < -1")
    if(R[2][2] == 0):
        beta = np.pi/2
    else:
        beta = np.arctan2((1-R[2][2]**2)**0.5, R[2][2])

    # gamma
    if(R[2][0] == 0.): # -c3s2 == 0
        if(beta == 0.): # 2D rotation about z of angle alpha + gamma
            gamma = np.arccos(R[0][0]) - alpha
        else:
            gamma = 0.
    else:
        gamma = np.arctan2(R[2][1], -R[2][0])
    return(alpha, beta, gamma)


