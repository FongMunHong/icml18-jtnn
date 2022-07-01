import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from nnutils import *
from chemutils import get_mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    # return map(lambda s: x == s, allowable_set)
    return [x == s for s in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def mol2graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        #mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            # atom features gets one hot encoding of atom attributes
            fatoms.append( atom_features(atom) )
            in_bonds.append([]) # in_bonds for that specific atom

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms # add total_atoms so that index does not overlap
            y = a2.GetIdx() + total_atoms

            # all bonds for one molecule [(-1, -1), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5), (6, 7), (7, 6), (7, 8), (8, 7), (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10), (10, 12), (12, 10), (12, 13), (13, 12), (13, 14), (14, 13), (14, 15), (15, 14), (15, 16), (16, 15), (16, 17), (17, 16), (16, 18), (18, 16), (9, 19), (19, 9), (19, 20), (20, 19), (4, 21), (21, 4), (21, 22), (22, 21), (22, 1), (1, 22), (20, 6), (6, 20), (18, 12)]
            # in_bonds for one molecule [[2], [1, 4, 45], [3, 6], [5, 8], [7, 10, 42], [9, 12], [11, 14, 47], [13, 16], [15, 18], [17, 20, 38], [19, 22, 24], [21], [23, 26], [25, 28], [27, 30], [29, 32], [31, 34, 36], [33], [35], [37, 40], [39, 48], [41, 44], [43, 46]]

            b = len(all_bonds) 
            all_bonds.append((x,y))
            # fatoms[x] to access one hot encoding of that atom
            # bond_features(bond) - bondtype + stereo prop one hot encoding
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) ) 
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        # print('smiles', smiles)
        # print('total atoms', get_mol(smiles).GetNumAtoms())
        # print('in_bonds', in_bonds)
        # print('fbonds', fbonds)
        # raise
        
        scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms


    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b

    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2


    # print('fatoms size', fatoms.size()) # total atoms, one hot encoding of atoms(39) 
    # print('fbonds size', fbonds.size()) # total bonds, one hot encoding of atoms + bonds
    # print('agraph size', agraph.size()) # total atoms of 40 trees, MAX_NB (6)
    # print('bgraph size', bgraph.size()) # total bonds of 40 trees, SIZE (6)
    # print('scope size', len(scope))

    return fatoms, fbonds, agraph, bgraph, scope

class MPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms,fbonds,agraph,bgraph,scope = mol_graph
        fatoms = create_var(fatoms) # atom only
        fbonds = create_var(fbonds) # atom + bond
        agraph = create_var(agraph) # agraph indexed by atoms 0 -> [1, 35, 44, 0, 0, 0]
        bgraph = create_var(bgraph)

        # x[uv] to indicate bond type
        binput = self.W_i(fbonds) # atom + bonds -- bond indexed (0, 1), (1, 0), (0, 2), (2, 0)
        message = nn.ReLU()(binput)

        # print('message', message)
        # print('message size', message.size())

        # two hidden vectors ν[uv] and ν[vu] denoting
        # the message from u to v and vice versa
        for i in range(self.depth - 1): # belief propagation happens here, each node aggregate info from its neighbors
            nei_message = index_select_ND(message, 0, bgraph) # bgraph index by bonds (0-1) -> [0, 35, 44, 0, 0, 0]
            # print('nei_message', nei_message)
            # print('nei_message size', nei_message.size())
            nei_message = nei_message.sum(dim=1)
            # print('nei_message size 2', nei_message.size())
            nei_message = self.W_h(nei_message)
            message = nn.ReLU()(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        # print('shape', mol_vecs.size()) # mol_vecs shape 40 x 450
        return mol_vecs

