from tkinter.tix import MAX
import torch
import torch.nn as nn
from collections import deque
from mol_tree import Vocab, MolTree
from nnutils import create_var, GRU

# max neighbors of a clique or atom?
MAX_NB = 8

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            # We pick an arbitrary leaf node as the root and propagate messages in two phases
            order = get_prop_order(root)
            orders.append(order)

        # for i, order in enumerate(orders[:1]):
        #     print(i , 'pair', '-root:', root_batch[i].idx)
        #     print('root neighbors', [nei.idx for nei in root_batch[i].neighbors])
        #     for pairs in order:
        #         pair = [tuple(node.idx for node in pair) for pair in pairs]
        #         print(pair, end=" | ")

        #     print('\n')

        # each C(i), C(j) has two message vectors m(ij), m(ji)
        # 0 pair -root: 0
        # root neighbors [10]
        # [(8, 12), (9, 12)] | [(12, 7)] | [(7, 11)] | [(11, 6)] | [(6, 5)] | [(5, 4)] 
        # | [(3, 13), (4, 13)] | [(13, 2)] | [(2, 1)] | [(1, 10)] | [(10, 0)] | [(0, 10)] | [(10, 1)] 
        # | [(1, 2)] | [(2, 13)] | [(13, 3), (13, 4)] | [(4, 5)] | [(5, 6)] | [(6, 11)] 
        # | [(11, 7)] | [(7, 12)] | [(12, 8), (12, 9)] |
                
        h = {}
        # out of the 40 separate trees, which one has the deepest depth or highest number of edges
        max_depth = max([len(x) for x in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)
        # print('padding', padding.size())

        # prop_list - propagation list
        for t in range(max_depth):
            prop_list = []
            # prop_list takes all 40 of ith order of trees and make them a list
            # A [ pairs1, pairs2, pairs3, pairs4,] mol tree node edges (clique pairs/fragment pairs)
            # B [ pairs1, pairs2, pairs3, pairs4,]
            # C [ pairs1, pairs2, pairs3, ]
            # prop_list = [A(pairs1), B(pairs1), C(pairs1)]
            # take note that pairs1 might consist several pair which are edges assiociated to message vector m(ij) , m(ji)
            # thus prop list will be list of these edges, and will not be 40 tree * 1 edge
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])
                    # print(len(order[t]), end="")

            cur_x = []
            cur_h_nei = []
            # print('len prop list', len(prop_list))
            # for each tree in batch 40, we have each pairs of cliques
            # in which the first node is the target node (cur_x)
            # and the we are calculating it's neighbors (cur_h_nei)
            for node_x,node_y in prop_list:
                # node x is the target node
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                # find all neighbors of node x
                # should be within the range of 8 (MAX_NB)
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue # one off, remove duplicate edges, ij, ji
                    h_nei.append(h[(z,x)]) # append embedding of all neighbors

                # if there is not enough neighbors, pad them  with 
                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            # get embedding h = num_of_nodes (batch size 40) x dimension of embedding (hidden size)
            cur_x = self.embedding(cur_x)
            # cur_h_nei will be (40 - 64), 8, 450 (prop list length, neighbors include padding , dimension of embedding)
            # cur_h_nei concatenate the whole rows of order[i] neighbors embedding, so we need to reshape the matrix
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            # print('cur_h_size', cur_h_nei.size())
            # update function GRU
            # takes the embedding of current node, with its neighbors 

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                # GRU is used aggregate messages to target node to create new message (new_h)
                h[(x,y)] = new_h[i]

        # h is just messages computed from neighbors
        # all of the messages computed will be aggregated to the root
        # different from the videos where they have all the nodes, and aggregate from the values of their neighbors
        # and the output is sum of the aggregation of all the nodes,
        # junction tree however only take the representation of the root into account
        # h(t) = h(root)

        # encoder  is used to compute zT , which only requires the bottom-up phase of the network
        # print('size h', len(h)) # size of entire ordering of 40 tree 
        # h is a dictionary where keys are edges, and values are message passing from Edge(x, y), neighbors of x,  
        # print('h', list(h.keys())[0])
        # print(h[list(h.keys())[0]])
        # print(h[list(h.keys())[0]].size())

        # h contains all 40 trees orders message, bottom up and top down message passing
        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs

"""
Helper functions
"""

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    # I believe this a BFS implementation, since there is no recursion
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    # nodes var are just root batches
    # node var is the root
    for node_x in nodes:
        x_idx.append(node_x.wid)
        # find neighbors of root, concatenate them together
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
        # print(torch.cat(nei, dim=0).view(-1, MAX_NB, hidden_size).tolist())
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    print('hnei', h_nei.size())
    # aggregating function for the root node, sum all the columns
    sum_h_nei = h_nei.sum(dim=1) # 40, 450 - root batch size (number of roots), hidden size 
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1) # the root itself and it's neighbors (40, 450) + (40, 450)
    # print(node_vec.size()) # size is (40, 900 [450 + 450])
    # transforming function for the root vector
    return nn.ReLU()(W(node_vec)) # [40 x 900 (.) 900 x 450] == [40 x 450]

