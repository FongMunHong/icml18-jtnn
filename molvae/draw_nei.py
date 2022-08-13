import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw

sys.path.append("../")
sys.path.append('../jtnn')

import numpy as np
from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
# model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=False)
# model.load_state_dict(torch.load(opts.model_path, map_location = torch.device('cpu')))
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

np.random.seed(0)
x = np.random.randn(latent_size)
x /= np.linalg.norm(x)

y = np.random.randn(latent_size)
y -= y.dot(x) * x
y /= np.linalg.norm(y)

# z0 = "CN1C(C2=CC(NC3C[C@H](C)C[C@@H](C)C3)=CN=C2)=NN=C1"
# z0 = "COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"
z0 = "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34" # not working
# z0 = "C1CC2CCC3CCC4CCC5CCC6CCC1C1C2C3C4C5C61" # not working
# z0 = "O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1" # not working
# z0 = "O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2" # works
# z0 = "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3" # works
# z0 = "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1" # not working
# z0 = "Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br" # works

z0 = model.encode_latent_mean([z0]).squeeze() # len z0 - (28 + 28) = 56
z0 = z0.data.cpu().numpy()

tree_z, mol_z = torch.Tensor(z0).unsqueeze(0).chunk(2, dim=1)
tree_z, mol_z = create_var(tree_z), create_var(mol_z)
s = model.decode(tree_z, mol_z, prob_decode=False)
m = Chem.MolFromSmiles(s)
# for bond in m.GetBonds():
#     print(bond.GetBondType())
# Chem.Kekulize(m)
# print("Ending", Chem.MolToSmiles(m,kekuleSmiles=True))
# raise

delta = 1
nei_mols = []
for dx in range(-6,7):
    for dy in range(-6,7):
        z = z0 + x * delta * dx + y * delta * dy
        tree_z, mol_z = torch.Tensor(z).unsqueeze(0).chunk(2, dim=1)
        tree_z, mol_z = create_var(tree_z), create_var(mol_z)
        # print(tree_z.size()) # size [1, 28]
        # print(mol_z.size()) # size [1, 28]
        nei_mols.append( model.decode(tree_z, mol_z, prob_decode=False) )
        raise

print(nei_mols)
nei_mols = [Chem.MolFromSmiles(s) for s in nei_mols]
img = Draw.MolsToGridImage(nei_mols, molsPerRow=13, subImgSize=(200,200), useSVG=True)
# print(img)
# img

