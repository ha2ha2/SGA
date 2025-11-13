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
#                     批量攻击执行                                  #
# ================================================================== #


# 存储攻击结果
attack_results = {
    'surrogate_before': [],  # 攻击前代理模型预测
    'surrogate_after': [],   # 攻击后代里模型预测  
    'target_before': [],     # 攻击前目标模型预测
    'target_after': [],      # 攻击后目标模型预测
    'success_surrogate': 0,  # 对代理模型攻击成功次数
    'success_target': 0,      # 对目标模型攻击成功次数
    'valid_count': 0      # 对目标模型攻击成功次数
}

print("Starting attacks on all target nodes...")
for target in tqdm(targets, desc="Attacking"):
    target = target.item()
    target_label = data.y[target].item()

    clean_sample_output_on_T = target_model.predict(data, mask=target)
    target_after_correct = (clean_sample_output_on_T.argmax() == target_label)
    if target_after_correct:
        attack_results['valid_count'] += 1
    else:
        continue
    
    # ================================================================== #
    #                      Attacking                                     #
    # ================================================================== #
    attacker = SGAttack(data, device=device)
    attacker.setup_surrogate(surrogate_model.model)
    attacker.reset()
    attacker.attack(target)
    
    # ================================================================== #
    #                      After evasion Attack                          #
    # ================================================================== #
    surrogate_model.cache_clear()        
    # ！！！！[注意！！！]在下面这行代码执行之前，不能使用attacker.data()，使用了的话，修改记录就没了，到这里就是原图，没变
    adv_output_on_S = surrogate_model.predict(attacker.data(), mask=target)           # 调用.data()函数的时候，data = copy(self.ori_data)是从原始raw graph上添加边的，之前的attack 过程已经把需要添加或者删除的边记录放进去了
    print("After evasion attack:")
    print(mark(adv_output_on_S, target_label))

    adv_output_on_T = target_model.predict(attacker.data(), mask=target)
    print("After evasion attack:")
    print(mark(adv_output_on_T, target_label))

    target_after_output = target_model.predict(attacker.data(), mask=target)
    
    # 检查预测是否正确 (与真实标签比较)
    target_after_correct = (target_after_output.argmax() == target_label)
    
    # ================================================================== #
    #                     统计结果                                     #
    # ================================================================== #
    attack_results['target_after'].append(target_after_correct)
    

    if target_after_correct:
        attack_results['success_target'] += 1

# ================================================================== #
#                     输出统计结果                                  #
# ================================================================== #
print("\n" + "="*50)
print("攻击结果统计")
print("="*50)

print(f"总攻击节点数: {len(targets)}")
print(f"总又掉节点数: {attack_results['valid_count']}")

# 计算准确率
surrogate_before_acc = np.mean(attack_results['surrogate_before'])
surrogate_after_acc = np.mean(attack_results['surrogate_after'])
target_before_acc = np.mean(attack_results['target_before'])
target_after_acc = np.mean(attack_results['target_after'])

print(f"\n准确率统计:")
print(f"代理模型 - 攻击前: {surrogate_before_acc:.2%}")
print(f"代理模型 - 攻击后: {surrogate_after_acc:.2%}")
print(f"目标模型 - 攻击前: {target_before_acc:.2%}")
print(f"目标模型 - 攻击后: {target_after_acc:.2%}")

print(f"\n攻击成功率:")
print(f"代理模型攻击成功率: {attack_results['success_surrogate']/len(targets):.2%}")
print(f"目标模型攻击成功率: {attack_results['success_target']/len(targets):.2%}")

# 详细统计
print(f"\n详细统计:")
print(f"代理模型 - 攻击前正确: {sum(attack_results['surrogate_before'])}/{len(targets)}")
print(f"代理模型 - 攻击后正确: {sum(attack_results['surrogate_after'])}/{len(targets)}")
print(f"目标模型 - 攻击前正确: {sum(attack_results['target_before'])}/{len(targets)}")
print(f"目标模型 - 攻击后正确: {sum(attack_results['target_after'])}/{len(targets)}")
print(f"代理模型攻击成功: {attack_results['success_surrogate']}/{len(targets)}")
print(f"目标模型攻击成功: {attack_results['success_target']}/{len(targets)}")

# 保存攻击结果
import os
if not osp.exists("../attack_results/"):
    os.makedirs("../attack_results/")

np.savez(f'../attack_results/{dataset}_SGA_results.npz', 
         targets=targets.cpu().numpy(),
         attack_results=attack_results)

print(f"\n攻击结果已保存到: ../attack_results/{dataset}_SGA_results.npz") 