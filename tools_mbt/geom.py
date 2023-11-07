#!/usr/bin/env python3

# Script for aligning two molecular structures
import numpy as np
from tools_mbt.misc import weights, atom_nums, atom_names

#weights    = {1: 1.0008, 6: 12.011, 7: 14.007, 8: 15.999}
#
#atom_nums  = {"H": 1, "He": 2, "Li": 3, "Be": 4, "Al": 5, "C": 6,\
#        "N": 7, "O": 8, "F": 9, "Ne": 10, "P": 15, 16: "S"}
#atom_names = {atom_nums[key]: key for key in atom_nums.keys()}

def quaternion_to_rot(q):
    """Get the 3x3 rotation matrix associated with the 
    input quaternion. q = (qr, qi, qj, qk)"""
    fac = 2./(np.linalg.norm(q)**2)
    print("fac = ", fac)
    d1 = 1 - fac*(q[2]**2 + q[3]**2)
    d2 = 1 - fac*(q[1]**2 + q[3]**2)
    d3 = 1 - fac*(q[1]**2 + q[2]**2)
    # Matrix elements multiplied by the pre-factor
    ri = fac*q[0]*q[1]
    rj = fac*q[0]*q[2]
    rk = fac*q[0]*q[3]
    ij = fac*q[1]*q[2]
    ik = fac*q[1]*q[3]
    jk = fac*q[2]*q[3]

    R = np.array([[d1,    ij - rk, ik + rj],
                  [ij + rk, d2,    jk - ri],
                  [ik - rj, jk + ri, d3   ]])
    return(R)

def translate_matrix(T, Mt):
    """Returns the target geometry (Mt) translated according to to the translation matrix T"""
    return(Mt + T)

def rotate_matrix(Rot, Mt):
    """Returns the target geometry (Mt) rotated according to to the rotation matrix Rot"""
    return(np.matmul(Mt, Rot))

class Align(object):
    """
       Align the Mtg target matrix to the Mref reference matrix

       Update: The rotation matrix must be computed at the origin, so both
       coordinate sets are temporarily translated. The rotation matrix
       self.R always acts on coordinates centered at the origin. However,
       the final aligned geometry provided is centered at the COM of the 
       reference geometry.
    """
    def __init__(self, Mref, Mtgt, atom_list):
        # Displace both geometres to the origin
        # before determining the rotation matrix
        self.cm_A   = np.average(Mref, axis = 0)
        self.cm_B   = np.average(Mtgt, axis = 0)
        self.A = Mref - self.cm_A
        self.B = Mtgt - self.cm_B
        
        self.dim = len(atom_list)
        self.atom_list = atom_list
        self.atom_weights = [weights[i] for i in self.atom_list]
        self.W = np.zeros((self.dim, self.dim))
        self.K = np.zeros((4,4))

        self.init_matrices()

    def init_matrices(self):
        # Both matrices are translated at the beginning
#        self.translate()
        self.get_W_matrix()
        self.get_M_matrix()
        self.get_K_matrix()
        self.get_optimal_rotation()

    def align(self):
        """It returns the target coordinates aligned to the reference coordinates"""
        R = self.rotate_target()
        return(R)
    
    # Displace COMs
    def translate(self):
        """Translate the center of mass of the target geometry to
           that of the reference geometry.
        """
#        print("Translate called") 
#        CD_A = np.average(self.A, axis = 0, weights = self.atom_weights)
#        print(CD_A)
#        CD_B = np.average(self.B, axis = 0, weights = self.atom_weights)
        CD_A = np.average(self.A, axis = 0)
        print(CD_A)
        CD_B = np.average(self.B, axis = 0)
        self.displ_vec = CD_A - CD_B
#        print("displ vec")
        print(self.displ_vec)
        displ = []
        for cnt, i in enumerate(self.B):
            displ.append(self.displ_vec) 
        self.displ = np.array(displ)
        self.B = self.B + self.displ 

    def get_translation(self):
        """Return displacement vector for target matrix"""
        return(self.displ_vec)

    def get_W_matrix(self):
        """Construct weights matrix. Arg contains a list with the atomic numbers
        of each atom"""
        self.W = np.identity(self.dim)
        for cnt, iat in enumerate(self.atom_list):
            self.W[cnt][cnt] = weights[iat]
    
    def get_M_matrix(self):
        """Construct scalar product matrix:
            M = (A+ * W * B))"""
        self.M = np.matmul(np.transpose(self.A), np.matmul(self.W, self.B))

    def get_K_matrix(self):
        """Construct the matrix K, whose highest eigenvalue determines the 
        optimal rotation matrix"""
        print("getting K")

        # Define explicitly
        
        diag = self.M.diagonal()
        d1 = diag[0] + diag[1] + diag[2]
        d2 = diag[0] - diag[1] - diag[2]
        d3 = -diag[0] + diag[1] - diag[2]
        d4 = -diag[0] - diag[1] + diag[2]
        M = self.M
        self.K = np.array([[d1, M[1][2] - M[2][1], M[2][0] - M[0][2], M[0][1] - M[1][0]],
                      [M[1][2] - M[2][1], d2, M[0][1] + M[1][0], M[2][0] + M[0][2]],
                      [M[2][0] - M[0][2], M[0][1] + M[1][0], d3, M[1][2] + M[2][1]],
                      [M[0][1] - M[1][0], M[2][0] + M[0][2], M[1][2] + M[2][1], d4]])
        
    def get_optimal_rotation(self):
        """Initializes the optimal rotation matrix R"""
        # w = eigenvalues, v = eigenvectors
        w, v = np.linalg.eig(self.K) 
        ind_max = int(np.where(w == w.max())[0])
        # The column selected corresponds to the eigenvector
        # of v.max()
        self.R_arr = v[:,ind_max].astype("float64") 
        self.R = quaternion_to_rot(self.R_arr) # Rotation matrix at the origin
    
    def get_rotation_matrix(self):
        """Return optimal rotation matrix"""
        return(self.R)

    def rotate_target(self):
        """
           Returns the target geometry aligned to the reference geometry
           Update: The target has been displaced to the origin. Displace it 
           to the COM of the reference to get the target geometry aligned
           to the reference
        """
        # Rotation matrix acts from the right
        aligned_tgt = np.matmul(self.B, self.R) + self.cm_A
        return(aligned_tgt)


