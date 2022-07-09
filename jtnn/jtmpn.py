import torch
import torch.nn as nn
from nnutils import create_var, index_select_ND
from chemutils import get_mol
#from mpn import atom_features, bond_features, ATOM_FDIM, BOND_FDIM
import rdkit.Chem as Chem

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    # return map(lambda s: x == s, allowable_set)
    return [x == s for s in allowable_set]


def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])

class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, cand_batch, tree_mess):
        fatoms,fbonds = [],[] 
        in_bonds,all_bonds = [],[] 
        mess_dict,all_mess = {},[create_var(torch.zeros(self.hidden_size))] #Ensure index 0 is vec(0)
        total_atoms = 0
        scope = []

        # tree message + candidate subgraph message
        # -------------------------------------------
        
        # print(len(tree_mess))

        # for e,vec in tree_mess.iteritems():
        # tree message saves embeddings for each edges, 
        # thus len of tree message corresponds to the number of edge traversals
        # total = []
        for e,vec in tree_mess.items():
            mess_dict[e] = len(all_mess) # mess_dict increments along the number of tree message
            all_mess.append(vec)
            # print(e, end="")
            # total.extend(list(e))

        # print(min(total)) # 0
        # print(max(total)) # depends on the last node of 40 trees being indexed
        # print(len(all_mess)) # depends on the number of traversals in total of the 40 trees

        for mol,all_nodes,ctr_node in cand_batch:
            # how they prepare the subgraph molecule with their respective index is important
            # because used here (enum_assemble)

            # mol - candidate subgraph generated 
            #       with neighbor of ctr_node and ctr_node

            # all_nodes - all nodes in that tree
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            # mol2 = Chem.Mol(mol)
            # for atom in mol2.GetAtoms():
            #     atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
            # print(Chem.MolToSmiles(mol2))
            # print(Chem.MolToSmiles(ctr_node.mol))
            # print(Chem.MolToSmiles(mol))

            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([]) # in_bonds for that specific atom
        
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms # get actual index, won't change despite of mapping

                # print('x, y', x, y)
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum() # based on mapping
                # print(x_nid, y_nid) # x_nid and b_nid is a product of enum_assemble, 
                # in which the whole fragment will be named the same number, 
                # thus the same number for both will refer being in the same MolTreeNode
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1
                # print(x_bid, y_bid)
                bfeature = bond_features(bond)
                # print(bfeature)

                b = len(all_mess) + len(all_bonds)  #bond idx offseted by len(all_mess)
                # print(len(all_mess),  len(all_bonds))
                all_bonds.append((x,y))
                # fatoms[x] to access one hot encoding of that atom
                # bfeature - bondtype + stereo prop one hot encoding
                fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                in_bonds[y].append(b)

                # print(in_bonds[y], y, b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                in_bonds[x].append(b)

                # print(in_bonds[x], x, b)
                # print()

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid: 
                    # if this two index on the candidate subgraph doesn't belong to the same MolTreeNode
                    # print('x_nid, y_nid', x_nid, y_nid) # idx on candidate subgraph - will show they don't belong to the same fragment
                    # print('x_bid, y_bid', x_bid, y_bid) # idx on molTree itself

                    if (x_bid,y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid,y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid,x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid,x_bid)]
                        in_bonds[x].append(mess_idx)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms

        
        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b # a refers to the bond index, i index refers to the values that needs to be filled up
                # [1072,    0,    0,    0,    0,    0,    0,    0,    0,    0], row value a = 0 (atom idx out of all 40 trees)
                #    i ,    i+1,  i+2,  i+3,
                # [1071, 1074, 1076,  456,    0,    0,    0,    0,    0,    0], row value a = 1 (atom idx out of all 40 trees)
                # print('atom_index', a, 'index on agraph to update', i, 'index of in_bonds that are associated to atom index a', b)
            # if a > 10: raise


        for b1 in range(total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by len(all_mess)
                if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
                    bgraph[b1,i] = b2
                    # print(b1, i, '+++', x, y) bonds associated for update
            # if b1 > 10: raise

        
        # print(agraph[:5])
        # tensor([[1072,    0,    0,    0,    0,    0,    0,    0,    0,    0],  -> 0 (atom_idx)
        #         [1071, 1074, 1076,  456,    0,    0,    0,    0,    0,    0],  -> 1
        #         [1073,    0,    0,    0,    0,    0,    0,    0,    0,    0],  -> 2
        #         [1075,  608,    0,    0,    0,    0,    0,    0,    0,    0],  -> 3
        #         [1078,    0,    0,    0,    0,    0,    0,    0,    0,    0]]) -> 4
        # print(bgraph[:5]) - each row corresponds to one bond, ignore first row, 0-1, 1-0, 0-2
        # tensor([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
        #         [   0, 1074, 1076,  456,    0,    0,    0,    0,    0,    0],  -> 0-1
        #         [1071,    0, 1076,  456,    0,    0,    0,    0,    0,    0],  -> 1-0
        #         [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  -> 1-2
        #         [1071, 1074,    0,  456,    0,    0,    0,    0,    0,    0]]) -> 1-3

        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        graph_message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message,graph_message], dim=0) # size 28349, 450
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = nn.ReLU()(binput + nei_message)

        message = torch.cat([tree_message,graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            # scope.append((total_atoms,n_atoms))
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0) # torch.Size([1341 (len cand batch), 450])
        # print(len(cand_batch), mol_vecs.size())
        return mol_vecs

