#!/usr/bin/env python3

import numpy as np
import re
import os
#from conversions import atomnum, ha2ev
from tools_mbt.misc import *
from tools_mbt.misc import trans_cart2sph as c2s
from tools_mbt.permutations import permute_orbitals

H_fact = 1.09
#ag2bohr = 1.8897259886

class Gamess_parser(object):

    def __init__(self, infile, charge = 0, mult = 1, order = 'pyscf', basis = 'spherical'):
        '''Generate a Mol object with all the molecular
           attributes retrieved from a GAMESS calculation.
           For the time being, retrieve data only from the last
           calculation (e.g.: a REKS)
        '''

        self.atomcoords  = []
        self.atomnames   = []
        self.atomnums    = []
        self.mullikenchg = [] # A (R,N) array,
                              # where N = #atoms
                              # and R = #roots
        self.espfchg     = [] # Similar to mullikenchg
                              # but with espf
        self.energies    = [] # A R-dim array
        self.extpot      = [] # External potential
        self.id_calc     = 0
      
        self.atomnos    = []
        self.order      = order # Ordering of AO. Default = pyscf 
        self.basis      = basis
        self.ncrt       = 0 # Number of cartesian basis functions
        self.nsph       = 0 # Number of real spherical basis functions
        self.nmo        = 0 # Number of MOs
        self.charge     = charge
        self.mult       = mult
        self.natoms     = 0

        # Input arrays for integrals
        self._bas = []
        self._atm = []
        self._env = [0.0 for i in range(0,20)]
        
        # Other
        self._pseudonorm = np.array([]) # Pseudonormalization factors

        # Parser for the different calculation types.
        # In this case the calculation type is given 
        # by the SCFTYP keyword in the $CONTRL section
        # For a given SCFTYP there are several different
        # options, e.g.: ROHF -> DFT -> TDDFT -> MRSF

        # This is important for parsing the energies
        self.dict_calc   = {'reks': '2SI-2SA-REKS(2,2)' }

        # Determine the calculation type and other
        # contrl properties
        self.contrl = {}

        with open(infile, 'r') as f:
            self.lines = f.read().splitlines()
        
#        # Get attributes
#        self.parse_contrl()
#        self.parse()
#        self.set_parser()

    def parse(self):
        '''Parse all arguments'''
        # Get attributes
        self.parse_contrl()
        self.parse_features()
        self.set_parser()


    def parse_contrl(self):
        '''Get important contrl settings'''
        for i, iline in enumerate(self.lines):
            if('$CONTRL OPTIONS' in iline):
                for j, jline in enumerate(self.lines[i+2:]):
                    if(len(jline) == 0): break
#                    entries = [ e for e in re.split('\W+', jline) if e]
                    entries = [ e.strip() for e in re.split('\W |=', jline) if e]
#                    print(entries)
                    for k in range(int(len(entries)/2)):
                        self.contrl[entries[2*k]] = entries[2*k+1]

    def set_parser(self):
        '''Set the parser attribute (class)'''
        if(self.contrl['SCFTYP'] == 'REKS'):
            self.parser = Parser_reks(self.lines)

            # Initialize attributes from the object parser
            self.haenergies = self.parser.haenergies
            self.energies   = self.parser.energies
            self.iroot      = self.parser.iroot
        if((self.contrl['SCFTYP'] == 'ROHF') and (self.contrl['TDDFT'] == 'MRSF')):
            self.parser = Parser_mrsf(self.lines)

            # Initialize attributes from the object parser
#            self.haenergies = self.parser.haenergies
            self.energies   = self.parser.energies
            self.iroot      = self.parser.iroot
        
        if(self.contrl['TDDFT'] == 'EXCITE'):
            self.parser = Parser_tddft(self.lines)

            # Initialize attributes from the object parser
#            self.haenergies = self.parser.haenergies
            self.energies   = self.parser.energies
            self.iroot      = self.parser.iroot
        if(self.contrl['SCFTYP'] == 'MCSCF'):
            self.parser = Parser_mcscf(self.lines)

            # Initialize attributes from the object parser
            self.haenergies = self.parser.haenergies
            self.energies   = self.parser.energies
            self.iroot      = self.parser.iroot

    def parse_features(self):
        '''Parse coordinates, 
                 atom names, 
                 atomic numbers, 
                 atomic charges
        '''
        # Get general molecular properties
        self.parse_general()
        # Get coordinates (Angs)
        self.parse_coords()
        # Get espf charges
        self.parse_espfchg()
        # Get external potential
        self.parse_extpot()
        # Get basis functions
        self.parse_shells()
        # Get MOs
        self.get_mos()

    def parse_general(self):
        '''Parse general molecular properties'''
        prop = ['nshls', # Total number of shells
                'ncrt',  # N. Cartesian basis funct.
                'nele',  # N. electrons
                'charge',#  
                'mult',  
                'amo',   # Alpha MOs
                'bmo',   # Beta MOs
                'natom', # N. atoms
                'qmatom',# N. QM atoms 
                'mmatom',# N. MM atoms
                'link']  # N. link atoms

        for i, iline in enumerate(self.lines):
            if('TOTAL NUMBER OF BASIS SET SHELLS' in iline):
                for j, jline in enumerate(self.lines[i:]):
                    if(len(jline) == 0): break
                    row = jline.split()
                    self[prop[j]] = int(row[-1])
                break

    def parse_coords(self):
        '''Parse cartesian coordinates of QM atoms (+ Link atom).
           If RUNTYP=OPTIMIZE, get last geometry only. Notice that in
           hybrid QM/MM calculations, atoms are ordered as 
           QM + Link + MM
        '''

        # Parse geometry from SP. Default units 
        # for coordinates: bohr
        if(self.contrl['RUNTYP'] == 'ENERGY' or self.contrl['RUNTYP'] == 'GRADIENT'):
            coord_parser = 'COORDINATES (BOHR)'
            offset = 2
            factor = 1
        elif(self.contrl['RUNTYP'] == 'OPTIMIZE'):
            coord_parser = '***** EQUILIBRIUM GEOMETRY LOCATED *****'
            offset = 4
            factor = ag2bohr # transform Angst to Bohr.

        for i, iline in enumerate(self.lines):
            if(coord_parser in iline):
                # Iterate over QM atoms only
                for atid, jline in enumerate(self.lines[i+offset:i+offset+self.qmatom]):
                    if(len(jline) == 0): break
                    row = jline.split()
                    atnum = atom_nums[row[0][0]]
                    ixyz = [factor * float(k) for k in row[2:5]]
                    self.atomnames.append(row[0])
                    self.atomnums.append(atnum)
                    self.atomnos.append(atnum)
                    self.atomcoords.append(ixyz)

                    self._atm.append([atnum, 20 + 4*atid, 1,\
                                     20 + 4*atid + 3, 0, 0])
                    self._env = self._env + ixyz + [0.0] # add "charge"
                break

        # Finally, get coordinates of link atoms, if any
        if(hasattr(self, 'link')):
            self._get_link_atoms()
            for ilink in self.link_atoms.keys():
                self.atomnames.append('HLA')
                self.atomnums.append(1)
                self.atomnos.append(1)
                ixyz = self.link_atoms[ilink][3].tolist()
                self.atomcoords.append([ag2bohr *xi for xi in ixyz])
                atid = len(self.atomcoords)
                self._env = self._env + ixyz + [0.0]
                self._atm.append([atnum, 20 + 4*atid, 1,\
                                 20 + 4*atid + 3, 0, 0])

        # Update arrays for integral calculations
        self.atomcoords = np.array(self.atomcoords).astype("float64")
        self._atm    = np.array(self._atm)
        self.natoms  = len(self.atomnos)

    def get_metadata_shells(self):
        """Get row data of basis functions from output file"""
        iatom = 0
        meta  = []
        for cnt1, iline in enumerate(self.lines):
            if("ATOMIC BASIS SET" in iline):
                for cn2, jline in enumerate(self.lines[cnt1 + 7:]):
                    row = jline.split()
                    if(len(row)>0):
                        if('TOTAL NUMBER OF BASIS SET SHELLS' in jline): break
            
                        elif((len(row) >= 1) and not (row[0].isdigit())):
                            # Iteration over atoms
                            iatom += 1
                        elif(row[0].isdigit() and len(row) == 5):
                            # Normal shell. Add column to the end
                            meta.append([str(iatom)] + row + ["0.0"])
                        elif(row[0].isdigit() and len(row) == 6):
                            # L shell
                            meta.append([str(iatom)] + row)
                break
        return(np.array(meta))

    def parse_shells(self):
        """
        bas: [[atom_idx, amom, n_prm, n_contr, 0, idx_frst_exp,
               idx_first_coeff, 0]..] -> ith shell
        atm: [[nucl_chg, id_x_crd, id+1 z_crd, 0, 0]]
        env: [0, ..., 0, ->0-19 idx
             x_crd_at1, y, z, 0.0, --> x, y, z , buff (chg?) atom 1
             x_crd_at2, y, z, 0.0,
             ...
             1_exp_1_shell, ..., last_exp_1_shell,
             1_cf_1_shell, ..., last_cf_1_shell,
             ...
             1_exp_last_shell_id, last_exp_last_shell_id,
             1_cf_last_shell_id, last_exp_last_shell_id]
    
        If the same primitives are used for different
        atoms, end saves the primitives only once
        and bas and atm point to these same positions
        """

        # Get pre-processed data
        meta = self.get_metadata_shells()
        # env buffer
        prm           = [] # [[exp1, ...,expn, cf1, ..., cfn], ...]
        # Iterate over atoms
        # Transform in int to keep ordering
        list_atoms = np.unique(meta[:,0].astype(int))
        for iatom in list_atoms:
            meta_atom = meta[meta[:,0] == str(iatom)]
            iatom_b = iatom - 1 # Indexing in _bas starting from 0
            # Iterate over shells
            shell_ids = np.unique(meta_atom[:,1].astype(int))
            for jshell in shell_ids:
                meta_shell = meta_atom[meta_atom[:,1] == str(jshell)]
                # Num. of Primitives per shell
                Nprim = len(meta_shell)
                # Ang. mom of shell
                amom_str = np.unique(meta_shell[:,2])[0].lower()

                # Assess whether this is a normal or an L shell
                if(amom_str == "l"):
                    # Save s and p shells separately
                    amom1      = 0
                    amom2      = 1
                    expn       = meta_shell[:,4].astype(float).tolist()
                    cf1_unnorm = meta_shell[:,5].astype(float).tolist()
                    cf2_unnorm = meta_shell[:,6].astype(float).tolist()
                    # Normalized coefficients
                    norm_cf1 = [gto_sph_norm(amom1, expn[i]) for i in range(len(cf1_unnorm))]
#                    cf1  = [gto_sph_norm(amom1, expn[i])*cf1_unnorm[i] for i in range(len(cf1_unnorm))]
                    cf1  = [cf1_unnorm[i] * norm_cf1[i] for i in range(len(cf1_unnorm))]
                    norm_cf2 = [gto_sph_norm(amom2, expn[i]) for i in range(len(cf2_unnorm))]
#                    cf2  = [gto_sph_norm(amom2, expn[i])*cf2_unnorm[i] for i in range(len(cf2_unnorm))]
                    cf2  = [cf2_unnorm[i] * norm_cf2[i] for i in range(len(cf2_unnorm))]
                    # Accumulate pseudonormalization coeff.
                    self._pseudonorm = np.concatenate((self._pseudonorm, norm_cf1 + norm_cf2))
                    
                    iprm = expn + cf1 + expn + cf2
                    if(iprm not in prm):
                        prm.append(iprm)
                        self._env = self._env + iprm
                        # Add s shell
                        idexp     = len(self._env) - 4*Nprim
                        idcf      = len(self._env) - 3*Nprim
                        self._bas.append([iatom_b, amom1, Nprim, 
                                          1, 0, idexp, idcf, 0])
                        # Add p shell
                        idexp     = len(self._env) - 2*Nprim
                        idcf      = len(self._env) - Nprim
                        self._bas.append([iatom_b, amom2, Nprim, 
                                          1, 0, idexp, idcf, 0])
                    else:
                        # Add s shell
                        natom = len(self.atomcoords)
                        id_prm = prm.index(iprm)
                        flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                        idexp = 20 + 4*natom + len(flat_prm)
                        idcf  = idexp + Nprim
                        self._bas.append([iatom_b, amom1, Nprim, 
                                          1, 0, idexp, idcf, 0])
                        # Add p shell
                        natom = len(self.atomcoords)
                        idexp = idexp + 2*Nprim
                        idcf  = idexp + Nprim
                        self._bas.append([iatom_b, amom2, Nprim, 
                                          1, 0, idexp, idcf, 0])

                else:
                    amom      = ang_mom[amom_str]
                    expn      = meta_shell[:,4].astype(float).tolist()
                    cf_unnorm = meta_shell[:,5].astype(float).tolist()
                    # Normalized coefficients
                    norm_cf   = [gto_sph_norm(amom, expn[i]) for i in range(len(cf_unnorm))]
#                    cf   = [gto_sph_norm(amom, expn[i])*cf_unnorm[i] for i in range(len(cf_unnorm))]
                    cf   = [cf_unnorm[i] * norm_cf[i] for i in range(len(cf_unnorm))]
                    # Accumulate pseudonormalization coeff.
                    self._pseudonorm = np.concatenate((self._pseudonorm, norm_cf))
                    
                    iprm = expn + cf
                    if(iprm not in prm):
                        prm.append(iprm)
                        self._env = self._env + iprm
                        idexp = len(self._env) - 2*Nprim
                        idcf  = len(self._env) - Nprim
                        self._bas.append([iatom_b, amom, Nprim, 1, 0, idexp, idcf, 0])

                    else:
                        # Primitives already present in _env
                        natom = len(self.atomcoords)
                        id_prm = prm.index(iprm)
                        flat_prm = [i for sublist in prm[:id_prm] for i in sublist]
                        idexp = 20 + 4*natom + len(flat_prm)
                        idcf  = idexp + Nprim
                        self._bas.append([iatom_b, amom, Nprim, 
                                          1, 0, idexp, idcf, 0])

        # Get number of basis functions
        self.get_num_ct()
        self.get_num_sp()

        # Re-define _bas and _env as numpy arrays
        self._bas = np.array(self._bas)
        self._env = np.array(self._env)
                
    def get_num_ct(self):
        """Get the number of cartesian functions from a bas array. No return type, but instead defines the attribute ncrt. Notice that ncrt is also defined on __int__"""
        self.ncrt = 0
        for cnt, ishl in enumerate(self._bas):
            self.ncrt += ang_mom_ct[ishl[1]]
    
    def get_num_sp(self):
        """Get the number of spherical functions from a bas array. No return type, but instead defines the attribute nsph"""
        self.nsph = 0
        for cnt, ishl in enumerate(self._bas):
            self.nsph += ang_mom_sp[ishl[1]]
                    
    # MO- referred methods
    def allocate_mo(self, unrestr = False):
        if(self.basis == "spherical"):
#            dim = get_num_sp(self._bas)
            dim = self.nsph
        elif(self.basis == "cartesian"):
#            dim = get_num_ct(self._bas)
            dim = self.ncrt
        else:
            raise ValueError("basis can only be either spherical or cartesian")
        if(unrestr == True):
            self.C_alpha = np.zeros((dim,dim))
            self.C_beta = np.zeros((dim,dim))
            self.C_mo = {"alpha": self.C_alpha,
                      "beta": self.C_beta}
        else:
            self.C_mo = np.zeros((dim,dim))
        # Number of MOs
        self.nmo = dim

    def _get_mo(self):
        """
        Parse molecular orbitals and MO energies.
        Rows = MO id
        cols = AO id
        """
        self.moenergies = []
        count_batch     = 0 # Count batches of 5 MOs
                            # (columns in .out)

        for cnt1, iline in enumerate(self.lines):
            if("EIGENVECTORS" in iline):
                for cnt2, jline in enumerate(self.lines[cnt1 + 4:]):
                    if("-----------" in jline): break
                    row = jline.split()
                    if(len(row) > 0):
                        is_header = is_float(row[0]) and not row[0].isdigit()
                        if(is_header):
                            # Row of energies found. Parse batch of MOs
                            self.moenergies = self.moenergies + [float(iene) for iene in row]
                            # Parse MO indices
                            id_mos = [int(imo) - 1 for imo in self.lines[cnt1 + 4 + cnt2 - 1].split()]
                            mos_in_batch = len(row)
                            offset = cnt1 + 4 + cnt2 + 2
                            for cnt3, kline in enumerate(self.lines[offset:offset + self.nmo]):
                                row_coef = [float(kelem) for kelem in kline.split()[4:]]
                                # Initialize matrix elements
                                for irow, icoef in enumerate(row_coef):
                                    self.C_mo[id_mos[irow]][cnt3] = icoef
                break
        self.moenergies = np.array(self.moenergies)

        # Permute AOs so as to follow the self.order
        # ordering
        perm_id = permute_orbitals(self._bas,\
                order = "gamess", target = self.order, basis = self.basis)
        self.C_mo = self.C_mo[:,perm_id]

    def get_mo_prop(self, unrestr = False):
        """Get MO properties such as
           spin and occupation numbers
        """
        self.mooccnos   = [] # MO occupation numbers
        self.mospin     = []
        if(not unrestr):
            # Restricted calculation
            self.mooccnos = np.array([2 for i in range(self.amo)] + [0 for i in range(self.amo, self.nmo)])
            self.mospin  = np.array(["Alpha" for i in range(self.nmo)])



    def get_all_mo(self, unrestr = False):
        """Get (for now restricted) MOs"""
        self._get_mo()

    def get_mos(self, unrestr = False):
        """Allocate MO arrays and retrieve corresponding values"""
        # Get MOs
        self.allocate_mo(unrestr)
        self.get_all_mo(unrestr)
        self.get_mo_prop(unrestr)

    def _get_link_atoms(self):
        '''Get coordinates of link atom. The coordinates are in Angstroem'''
        self.link_atoms = {} # {number : [QM, MM, factor, [x,y,z]]}

        # Get linking atom pair indices and factor
        for i, iline in enumerate(self.lines):
            if('LINKING ATOM PAIRS' in iline):
                for j, jline in enumerate(self.lines[i+3:i+3+self.link]):
                    row = jline.split()
                    self.link_atoms[int(row[0])] = [int(row[1]), int(row[2]), float(row[3])]
                    
        # Get coordinates of the two boundary atoms
        parser_coord = 'Cartesian Coordinates of Atoms in Bulk Model (ANGS)'
        indices = [k for k, kline in enumerate(self.lines) if parser_coord in kline]
        id_last = indices[-1]
        
        # Iterate over link atoms, 
        # then iterate over all atom coordinates
        for ilink in self.link_atoms.keys():

            crd_list  = []
            link_list = []
            # Search QM atom, then MM atom
            for i in range(0,2):
               found = False
               for j, jline in enumerate(self.lines[id_last+3:]):
                    if(found): break
                    row = jline.split()
                    if(int(row[0]) == self.link_atoms[ilink][i]):
                        crd_list.append([float(crd) for crd in row[2:5]])
                        found = True
            
#            print(crd_list)
            # Get unit vector between atoms
            crd_list = np.array(crd_list)
            d = crd_list[1] - crd_list[0] # MM - QM
            d = d/np.linalg.norm(d)
            # Finally, determine link atom coordinates
            # along he vector d
            l = crd_list[0] + d * H_fact

            # Append link atom coordinates to self.link_atom[ilink]
            self.link_atoms[ilink].append(l)

    def parse_espfchg(self):
        '''Get ESPF point charges'''
        if(self.contrl['RUNTYP'] == 'ENERGY' or self.contrl['RUNTYP'] == 'GRADIENT'):
            espf_parser = 'ESPF POPULATION ANALYSIS'
            offset = 5
        elif(self.contrl['RUNTYP'] == 'OPTIMIZE'):
            espf_parser = '--- ESPF POPULATION ANALYSIS ---'
            offset = 2
        for i, iline in enumerate(self.lines):
            if(espf_parser in iline):
                for jline in self.lines[i+offset:]:
                    if((len(jline) == 0) or ("$" in jline) or (len(iline) !=4)): break
                    row = jline.split()
                    self.espfchg.append(float(row[3]))
                break
    
    def parse_extpot(self):
        '''Parse external potential'''
        extpot_parser = 'External potential:'
        offset = 1
        for i, iline in enumerate(self.lines):
            if(extpot_parser in iline):
                for jline in self.lines[i+offset:]:
                    row = jline.split()
                    if((len(jline) == 0) or len(row) == 0): break
                    self.extpot.append(float(row[3]))
                break
        linkpot_parser = 'External potential on link atoms:'
        offset = 1
        # Determine external potential on link atoms, if present
        for i, iline in enumerate(self.lines):
            if(linkpot_parser in iline):
                for jline in self.lines[i+offset:]:
                    row = jline.split()
                    if((len(jline) == 0) or len(row) == 0): break
                    self.extpot.append(float(row[3]))
                break

    def __getattribute__(self, name):
        return(object.__getattribute__(self, name))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return(object.__getattribute__(self, name))


class Parser_reks(object):

    def __init__(self, lines):
        '''Class for REKS parser'''

        self.lines = lines

        self.haenergies  = [] # Abs. Energy in Ha
        self.energies    = [] # Relative Energy in eV
        self.id_step     = 0  # Index to track the last
                              # step of a geometry opt.
        self.iroot       = 1  # Relax root

        self.parse_general()

    def parse_general(self):
        '''Parse energies'''

        parser_ene = '2SI-2SA-REKS(2,2)' 

        # Get all indices with energy entry
        indices = [i for i, iline in enumerate(self.lines) if parser_ene in iline]
        if(len(indices)>0):
            self.id_step = indices[-1]
        else:
            raise(ValueError(parse_ene + ' not found'))
        
        # Get S0 and S1 energies
        s0 = float(self.lines[self.id_step + 3].split()[3])
        s1 = float(self.lines[self.id_step + 4].split()[3])
        self.haenergies.append(s0)
        self.haenergies.append(s1)
        self.energies.append(0)
        self.energies.append(ha2ev*(s1-s0))

        # Get root
        for i, iline in enumerate(self.lines):
            if('set rexTarget to') in iline:
                self.iroot = int(iline.split()[4])
                break

class Parser_mrsf(object):

    def __init__(self, lines):
        '''Class for MRSF parser'''

        self.lines = lines

        self.haenergies  = [] # Abs. Energy in Ha
        self.energies    = [] # Relative Energy in eV
        self.id_step     = 0  # Index to track the last
                              # step of a geometry opt.
        self.iroot       = 1  # Relax root

        self.parse_general()

    def parse_general(self):
        '''Parse energies'''

        parser_ene = 'TRANSITION        EXCITATION' 

        # Get all indices with energy entry
        indices = [i for i, iline in enumerate(self.lines) if parser_ene in iline]
        if(len(indices)>0):
            self.id_step = indices[-1]
        else:
            raise(ValueError(parse_ene + ' not found'))
        
#        # Get S0 and S1 energies
#        s0 = float(self.lines[self.id_step + 3].split()[3])
#        s1 = float(self.lines[self.id_step + 4].split()[3])
#        self.haenergies.append(s0)
#        self.haenergies.append(s1)
    
        self.energies.append(0)
        for i, iline in enumerate(self.lines[self.id_step+3:]):
            row = iline.split()
            if(len(row) == 0): break
#            print(row)
            self.energies.append(float(row[3]))

        # Get root
        for i, iline in enumerate(self.lines):
            if('IROOT =') in iline:
                self.iroot = int(iline.split()[4])
                break

class Parser_tddft(object):

    def __init__(self, lines):
        '''Class for MRSF parser'''

        self.lines = lines

        self.haenergies  = [] # Abs. Energy in Ha
        self.energies    = [] # Relative Energy in eV
        self.id_step     = 0  # Index to track the last
                              # step of a geometry opt.
        self.iroot       = 1  # Relax root

        self.parse_general()

    def parse_general(self):
        '''Parse energies'''

        parser_ene = 'STATE             ENERGY     EXCITATION' 

        # Get all indices with energy entry
        indices = [i for i, iline in enumerate(self.lines) if parser_ene in iline]
        if(len(indices)>0):
            self.id_step = indices[-1]
        else:
            raise(ValueError(parse_ene + ' not found'))
        
#        self.energies.append(0)
        for i, iline in enumerate(self.lines[self.id_step+2:]):
            row = iline.split()
            if(len(row) == 0): break
#            print(row)
            self.energies.append(float(row[3]))

        # Get root
        for i, iline in enumerate(self.lines):
            if('IROOT =') in iline:
                self.iroot = int(iline.split()[4])
                break


class Parser_mcscf(object):

    def __init__(self, lines):
        '''Class for MCSCF parser'''

        self.lines = lines

        self.haenergies  = [] # Abs. Energy in Ha
        self.energies    = [] # Relative Energy in eV
        self.id_step     = 0  # Index to track the last
                              # step of a geometry opt.
        self.iroot       = 1  # Relax root

        self.parse_general()

    def parse_general(self):
        '''Parse SA-energies'''

        parser_ene = 'THE DENSITIES ARE STATE AVERAGED OVER'

        # Get energies
        for i, iline in enumerate(self.lines):
            if(parser_ene in iline):
                for jline in self.lines[i+1:]:
                    if('SIEVING THE' in jline): break
                    row = jline.split()
                    self.haenergies.append(float(row[3]))
                break
        self.energies = [ha2ev * (self.haenergies[i] - \
                                  self.haenergies[0]) \
                                  for i in range(len(self.haenergies))]

        # Get root
        for i, iline in enumerate(self.lines):
            if('IROOT =') in iline:
                self.iroot = int(iline.split()[4])
                break


if(__name__ == '__main__'):
    from argparse import ArgumentParser as ap

    parser = ap('GAMEES output file parser')
    parser.add_argument('-f', dest = 'infile', type = str,\
            help = 'GAMESS output file')
    options = parser.parse_args()
    infile = options.infile
#    mol = Mol(infile) 
    mol = Gamess_parser(infile) 
