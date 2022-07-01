import torch
import torch.nn as nn
from mol_tree import Vocab, MolTree, MolTreeNode
from nnutils import create_var, GRU
from chemutils import enum_assemble
import copy

MAX_NB = 8
MAX_DECODE_LEN = 100

class JTNNDecoder(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # print("lt and hidden ", latent_size + hidden_size)
        # print("hidden_size", hidden_size)
        #Feature Aggregate Weights
        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)

        #Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.idx = -1
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x,y,z in trace]
       
    def forward(self, mol_batch, mol_vec):
        super_root = MolTreeNode("")
        super_root.idx = -1

        #Initialize
        pred_hiddens,pred_mol_vecs,pred_targets = [],[],[]
        stop_hiddens,stop_targets = [],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], super_root)

            # s_wid = [[node.wid if 'recover' in dir(node) else node for node in clique] for clique in s]
            # s_idx = [[node.idx if 'recover' in dir(node) else node for node in clique] for clique in s]
            # print(s_wid)
            # print()
            # print(s_idx)

            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = [] # reset neighbors


        # print('mol_batch size', len(mol_batch)) # 40 size
        #Predict Root
        pred_hiddens.append(create_var(torch.zeros(len(mol_batch),self.hidden_size))) # 40 x 450
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch]) # length 40
        pred_mol_vecs.append(mol_vec) # mol_vec is the vector after GRU and message passing on junction tree [[40, 450]]

        max_iter = max([len(tr) for tr in traces]) # find longest tree traversal
        padding = create_var(torch.zeros(self.hidden_size), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t]) # [MolTreeNode1, MolTreeNode2, 1 or 0] depending on value
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]

            # print('len prop_list', len(prop_list)) # 40 - batch size

            for node_x,real_y,_ in prop_list: # [MolTreeNode1, MolTreeNode2, 1 or 0]
                #Neighbors for message passing (target not included)
                # get message of neighbors of all node_x except direct neighbor y specified by edge (x, y)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx] # empty on first iteration
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors] # get all neighboring messages
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len) # not enough neighbors pad them with zeros

                #Current clique embedding
                cur_x.append(node_x.wid)


            #Clique embedding
            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x) # get embedding of ith (1st, 2nd, 3rd..) node of all trees

            #Message passing
            # print(torch.stack(cur_h_nei, dim=0).size()) # (8 neighbors.40 tree = 320) x 450 hidden size
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            # print(cur_h_nei.size()) # 40, 8, 450
            
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h) # cur_h_nei [40, 8, 450] -> new_h [40, 450], summarize contents from neighbors
            # print(new_h)
            # print(new_h.size())

            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size) # 320 x 450 -> 40 x 8 x 450
            cur_o = cur_o_nei.sum(dim=1)
            # print(cur_o.size()) # 320 x 450

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1: # top down = 1, bottom up = 0, might be otherwise
                    pred_target.append(node_y.wid)
                    pred_list.append(i) # top down edge traversal list [MolTreeNode1, MolTreeNode2, 1], [MolTreeNode3, MolTreeNode4, 1]
                stop_target.append(direction)

            # print('pred_target', pred_target) # [523, 521, 469, 343, 516, 515, 469, 517, 515, 515, 469, 469, 515, 469, 523, 469, 515, 515, 213, 469, 213, 479, 469, 515, 250, 449, 515, 469, 469, 517, 213, 469, 469, 482, 188, 726, 515, 469, 515, 515]
            # print('pred_list', pred_list) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
            # print('stop_target', stop_target) # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            #Hidden states for stop prediction
            # print(batch_list) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
            cur_batch = create_var(torch.LongTensor(batch_list))
            cur_mol_vec = mol_vec.index_select(0, cur_batch) # mol_vec is the vector after GRU and message passing on junction tree [[40, 450]]

            # print(cur_x.size())
            # print(cur_o.size())
            # print(cur_batch.size())
            # print(cur_mol_vec.size())
            # print(mol_vec)
            # print(mol_vec.size())
            # print()

            stop_hidden = torch.cat([cur_x,cur_o,cur_mol_vec], dim=1) # cur_x - embedding of nodes, cur_o - message from neighbors, # predicted message vector from encoder 
            # 40 x 928 [450 + 450 + 28(became small after sampling)] stop_hidden
            # print(stop_hidden.size())
            stop_hiddens.append( stop_hidden )
            stop_targets.extend( stop_target )
            # print(stop_targets)
            # raise
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_mol_vecs.append( mol_vec.index_select(0, cur_batch) )

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x,cur_o,mol_vec], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_mol_vecs = torch.cat(pred_mol_vecs, dim=0)
        pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_targets = create_var(torch.LongTensor(pred_targets))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_vecs = nn.ReLU()(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()
        stop_targets = create_var(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()
    
    def decode(self, mol_vec, prob_decode):
        stack,trace = [],[]
        init_hidden = create_var(torch.zeros(1,self.hidden_size))
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))

        #Root Prediction
        root_hidden = torch.cat([init_hidden, mol_vec], dim=1)
        root_hidden = nn.ReLU()(self.W(root_hidden))
        root_score = self.W_o(root_hidden)
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, self.vocab.get_slots(root.wid)) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x,fa_slot = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(torch.LongTensor([node_x.wid]))
            cur_x = self.embedding(cur_x)

            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hidden = torch.cat([cur_x,cur_h,mol_vec], dim=1)
            stop_hidden = nn.ReLU()(self.U(stop_hidden))
            stop_score = nn.Sigmoid()(self.U_s(stop_hidden) * 20).squeeze()
            
            if prob_decode:
                backtrack = (torch.bernoulli(1.0 - stop_score.data)[0] == 1)
            else:
                backtrack = (stop_score.item() < 0.5)

            if not backtrack: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_hidden = torch.cat([new_h,mol_vec], dim=1)
                pred_hidden = nn.ReLU()(self.W(pred_hidden))
                pred_score = nn.Softmax()(self.W_o(pred_hidden) * 20)
                if prob_decode:
                    sort_wid = torch.multinomial(pred_score.data.squeeze(), 5)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = step + 1
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            if backtrack: #Backtrack, use if instead of else
                if len(stack) == 1: 
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

"""
Helper Functions:
"""

def dfs(stack, x, fa):
    for y in x.neighbors:
        if y.idx == fa.idx:
            continue
        stack.append((x,y,1))
        dfs(stack, y, x)
        stack.append((y,x,0))

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True
    
def can_assemble(node_x, node_y):
    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

