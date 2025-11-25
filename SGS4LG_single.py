import os.path as osp
from lrgae.dataset import load_dataset

import torch
import torch_geometric.transforms as T

from greatx.attack.targeted import SGAttack
from greatx.datasets import GraphDataset
from greatx.nn.models import SGC, GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import mark, split_nodes

# torch.cuda.set_device(0)
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
# device = torch.device('cpu')
# ================================================================== #
#                     Attack Setting                                 #
# ================================================================== #
target = 153749  # target node to attack
target_label = data.y[target].item()

# ================================================================== #
#                      Load Model                                    #
# ================================================================== #
surrogate_model = Trainer(SGC(num_features, num_classes), device=device)
surrogate_model_cpkpath = './models_weights/SGC.pth'
surrogate_model.model.load_state_dict(torch.load(surrogate_model_cpkpath))

print('On Local Surrogate Model.....')
print('Testing.....')
test_res = surrogate_model.test_step(data, mask=splits.test_nodes)
print(test_res)

target_model = Trainer(GCN(num_features, num_classes), device=device)
target_model_cpkpath = './models_weights/GCN.pth'
target_model.model.load_state_dict(torch.load(target_model_cpkpath))

print('On Remote TARGET model.....')
print('Testing.....')
test_res = target_model.test_step(data, mask=splits.test_nodes)
print(test_res)

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = SGAttack(data, device=device)
attacker.setup_surrogate(surrogate_model.model)
attacker.reset()            # pubmed 1150M
attacker.attack(target)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
surrogate_model.cache_clear()        
# ！！！！[注意！！！]在下面这行代码执行之前，不能使用attacker.data()，使用了的话，修改记录就没了，到这里就是原图，没变
output = surrogate_model.predict(attacker.data(), mask=target)           # 调用.data()函数的时候，data = copy(self.ori_data)是从原始raw graph上添加边的，之前的attack 过程已经把需要添加或者删除的边记录放进去了
print("After evasion attack:")
print(mark(output, target_label))
# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
output = target_model.predict(attacker.data(), mask=target)
print("After evasion attack:")
print(mark(output, target_label))
