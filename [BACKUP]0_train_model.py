import os.path as osp
from lrgae.dataset import load_dataset

import torch
import torch_geometric.transforms as T
from greatx.utils import set_seed
from greatx.attack.targeted import SGAttack
from greatx.datasets import GraphDataset
from greatx.nn.models import SGC, GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import mark, split_nodes, set_seed
import numpy as np
from tqdm import tqdm

set_seed(42)
dataset = 'Reddit'
root = '/data1/home/ha2/dataset/pyg_data' 
# root = '/root/autodl-tmp/dataset/pyg_data'
print(GraphDataset.available_datasets())
transform=T.LargestConnectedComponents()
data = load_dataset(root, dataset, transform=transform)
print(data)

splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================== #
#                     Attack Setting                                 #
# ================================================================== #
targets = splits.test_nodes[torch.randperm(len(splits.test_nodes))[:1000]]
print(f"Randomly Selected {len(targets)} target nodes from test set")

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
surrogate_model = Trainer(SGC(num_features, num_classes), device=device, lr=0.1,
                         weight_decay=1e-5)
ckp = ModelCheckpoint('surrogate_model.pth', monitor='val_acc')
surrogate_model.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp])
torch.save(surrogate_model.model.state_dict(), './models_weights/SGC.pth')
print('On Local Surrogate Model.....')
print('Testing.....')
test_res = surrogate_model.test_step(data, mask=splits.test_nodes)
print(test_res)

target_model = Trainer(GCN(num_features, num_classes), device=device, lr=0.1,
                         weight_decay=1e-5)
ckp1 = ModelCheckpoint('target_model.pth', monitor='val_acc')
target_model.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp1])
torch.save(target_model.model.state_dict(), './models_weights/GCN.pth')
print('On Remote TARGET model.....')
print('Testing.....')
test_res = target_model.test_step(data, mask=splits.test_nodes)
print(test_res)




"""
['cora_full', 'blogcatalog', 'flickr', 'acm', 'polblogs', 'citeseer_full', 'pdn', 'cora_ml', 'coauthor_phy', 'uai', 'citeseer', 'amazon_cs', 'karate_club', 'amazon_photo', 'dblp', 'pubmed', 'coauthor_cs', 'cora']
Data(x=[232965, 602], edge_index=[2, 114615892], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965])
Randomly Selected 1000 target nodes from test set
Received extra configuration:
╒══════════════╤═══════════╕
│ Names        │   Objects │
╞══════════════╪═══════════╡
│ lr           │     0.1   │
├──────────────┼───────────┤
│ weight_decay │     1e-05 │
╘══════════════╧═══════════╛
Training...
100/100 [====================] - Total: 1.20s - 12ms/step- loss: 0.187 - acc: 0.957 - val_loss: 0.325 - val_acc: 0.94
On Local Surrogate Model.....
Testing.....
{'loss': 0.328948438167572, 'acc': 0.9398031234741211}
Received extra configuration:
╒══════════════╤═══════════╕
│ Names        │   Objects │
╞══════════════╪═══════════╡
│ lr           │     0.1   │
├──────────────┼───────────┤
│ weight_decay │     1e-05 │
╘══════════════╧═══════════╛
Training...
100/100 [====================] - Total: 50.83s - 508ms/step- loss: 0.292 - acc: 0.93 - val_loss: 0.332 - val_acc: 0.935
On Remote TARGET model.....
Testing.....
{'loss': 0.32717475295066833, 'acc': 0.9330907464027405}
"""