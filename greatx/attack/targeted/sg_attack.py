from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm.auto import tqdm

from greatx.attack.targeted.targeted_attacker import TargetedAttacker
from greatx.nn.models.surrogate import Surrogate
from greatx.utils import ego_graph

SubGraph = namedtuple('SubGraph', [
    'edge_index', 'sub_edges', 'non_edges', 'edge_weight', 'non_edge_weight',
    'selfloop_weight'
])


class SGAttack(TargetedAttacker, Surrogate):
    r"""Implementation of `SGA` attack from the:
    `"Adversarial Attack on Large Scale Graph"
    <https://arxiv.org/abs/2009.03488>`_ paper (TKDE'21)

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be :obj:`__class__.__name__`,
        by default None
    kwargs : additional arguments of :class:`greatx.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`

    Example
    -------
    .. code-block:: python

        from greatx.dataset import GraphDataset
        import torch_geometric.transforms as T

        dataset = GraphDataset(root='.', name='Cora',
                                transform=T.LargestConnectedComponents())
        data = dataset[0]

        surrogate_model = ... # train your surrogate model

        from greatx.attack.targeted import SGAttack
        attacker = SGAttack(data)
        attacker.setup_surrogate(surrogate_model)
        attacker.reset()
        # attacking target node `1` with default budget set as node degree
        attacker.attack(target=1)

        # attacking target node `1` with budget set as 1
        attacker.attack(target=1, num_budgets=1)

        attacker.data() # get attacked graph

        attacker.edge_flips() # get edge flips after attack

        attacker.added_edges() # get added edges after attack

        attacker.removed_edges() # get removed edges after attack

    Note
    ----
    * `SGAttack` is a scalable attack that can be applied to large scale graph
    * Please remember to call :meth:`reset` before each attack.
    """

    # SGAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton = True

    @torch.no_grad()
    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        *,
        tau: float = 5.0,
        freeze: bool = True,
    ):

        Surrogate.setup_surrogate(self, surrogate=surrogate, tau=tau,
                                  freeze=freeze)

        self.logits = self.surrogate(self.feat, self.edge_index,
                                     self.edge_weight).cpu()

        return self

    def set_normalize(self, state):
        # TODO: this is incorrect for models
        # with `normalize=False` by default
        for layer in self.surrogate.modules():
            if hasattr(layer, 'normalize'):
                layer.normalize = state
            if hasattr(layer, 'add_self_loops'):
                layer.add_self_loops = state

    def strongest_wrong_class(self, target, target_label):
        logit = self.logits[target].clone()
        logit[target_label] = -1e4
        return logit.argmax()

    def get_subgraph(self, target, target_label, best_wrong_label):
        sub_nodes, sub_edges = ego_graph(self.adjacency_matrix, int(target),            # construct a k-hop subgraph G(sub) = (A(sub), X)
                                         self.K)
        if sub_edges.size == 0:
            raise RuntimeError(
                f"The target node {int(target)} is a singleton node.")
        sub_nodes = torch.as_tensor(sub_nodes, dtype=torch.long,
                                    device=self.device)
        sub_edges = torch.as_tensor(sub_edges, dtype=torch.long,
                                    device=self.device)
        attacker_nodes = torch.where(
            self.label == best_wrong_label)[0].cpu().numpy()
        neighbors = self.adjacency_matrix[target].indices       
        attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)            # 从attacker_nodes中移除所有出现在neighbors(直接和target node相连的节点)中的节点 
        
        # 返回的子图：k跳子图 的顶点 并 all attacker_nodes的顶点
        # influencer → attacker 的潜在边 + k跳子图的边（已有边） + influencer → attacker 的潜在边的反向边 + k跳子图的边（已有边）的反向边 + 自环边
        influencers = [target]       
        subgraph = self.subgraph_processing(sub_nodes, sub_edges, influencers,
                                            attacker_nodes)

        if self.direct_attack:
            influencers = [target]
            num_attackers = self.num_budgets + 1        # 为什么+1？
        else:
            influencers = neighbors
            num_attackers = 3
        #  eventually leave ∆ potential nodes Vˆp ⊆ Vp, where the gradients of potential edges between t and them are ∆-largest   
        attacker_nodes = self.get_top_attackers(subgraph, target, target_label,
                                                best_wrong_label,
                                                num_attackers=num_attackers)
        # attacker_nodes进一步缩小，是原本attacker_modes中损失函数反向传播梯度top-k大的边对应的节点

        # 返回的子图：k跳子图 的顶点 并 top-k attacker_nodes的顶点
        # influencer → attacker 的潜在边 + k跳子图的边（已有边） + influencer → attacker 的潜在边的反向边 + k跳子图的边（已有边）的反向边  + 自环边
        
        subgraph = self.subgraph_processing(sub_nodes, sub_edges, influencers,
                                            attacker_nodes)

        return subgraph

    def get_top_attackers(self, subgraph, target, target_label,
                          best_wrong_label, num_attackers):
        torch.cuda.reset_peak_memory_stats()
        non_edge_grad, _ = self.compute_gradients(subgraph, target,
                                                  target_label,
                                                  best_wrong_label)
        print(torch.cuda.max_memory_allocated() / 1e9, "GB")
        _, index = torch.topk(non_edge_grad, k=min(num_attackers,           #取 "期望攻击者数量" 和 "实际候选边数量" 的较小值
                                                   non_edge_grad.size(0)),
                              sorted=False)
        attacker_nodes = subgraph.non_edges[1][index]   # 取这些边(index)的目标节点
        return attacker_nodes.tolist()

    def subgraph_processing(self, sub_nodes, sub_edges, influencers,
                            attacker_nodes):
        row = np.repeat(influencers, len(attacker_nodes))   # 这里的attacker_nodes是从attacker_nodes中移除所有出现在neighbors(直接和target node相连的节点)中的节点 
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])        # 潜在边（能添加的边）non_edges.shape [2, len(attacker_nodes)]  和attack_nodes
        """
        直接攻击的 non_edges：
        包含所有可能的 influencer → attacker 边
        不管这些边在原始图中是否已存在
        可能包含重复边

        间接攻击的 non_edges：
        只包含原始图中不存在的 influencer → attacker 边
        过滤掉了已存在的边
        """### 过滤已有边
        if not self.direct_attack:  # indirect attack        如果influencer是 N(t)，那这里的mask就有作用了，要把N(t) --> attack 本身就已经有边的去掉，不能算进潜在边。因为是对潜在边做添加边操作，对已有边做删除操作。所以都已经有边了还做什么添加边操作，这里是要过滤掉已有边的  
            mask = self.adjacency_matrix[non_edges[0], non_edges[1]].A1 == 0        
            non_edges = non_edges[:, mask]

        non_edges = torch.as_tensor(non_edges, dtype=torch.long,
                                    device=self.device)
        attacker_nodes = torch.as_tensor(attacker_nodes, dtype=torch.long,
                                         device=self.device)
        selfloop = torch.unique(torch.cat([sub_nodes, attacker_nodes])) # 为子图所有节点添加自环边
        test_value = 33487150
        """
        如果边数过多，会抛出下面这个错误  33487100是本地台式机的临界值
            return grad(loss,   # 要微分的标量张量（损失函数）
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            File "/data1/home/ha2/anaconda3/envs/SGA/lib/python3.12/site-packages/torch/autograd/__init__.py", line 412, in grad
                result = _engine_run_backward(
                        ^^^^^^^^^^^^^^^^^^^^^
            File "/data1/home/ha2/anaconda3/envs/SGA/lib/python3.12/site-packages/torch/autograd/graph.py", line 744, in _engine_run_backward
                return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            RuntimeError: CUDA error: an illegal memory access was encountered
            CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
            For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
            Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

        """
   
        total = 34887070
        perm = torch.randperm(total, device=non_edges.device)
        selected = perm[:test_value]
        edge_index = torch.cat([
            non_edges[:,:test_value], sub_edges[:,selected],       
            non_edges.flip(0)[:,:test_value],      # 反向候选边
            sub_edges.flip(0)[:,selected],      # 反向原始边
            selfloop.repeat((2, 1)) # 自环边
        ], dim=1)

        edge_weight = torch.ones(sub_edges.size(1),
                                 device=self.device).requires_grad_()
        non_edge_weight = torch.zeros(non_edges.size(1),
                                      device=self.device).requires_grad_()
        selfloop_weight = torch.ones(selfloop.size(0), device=self.device)

        subgraph = SubGraph(
            edge_index=edge_index,              # 包含所有边（正向+反向+自环）
            sub_edges=sub_edges[:,selected],
            non_edges=non_edges[:,:test_value],
            edge_weight=edge_weight[:test_value],            # 边权重为1
            non_edge_weight=non_edge_weight[:test_value],    # 边权重为0
            selfloop_weight=selfloop_weight,    # 边权重为1
        )
        return subgraph

    def attack(self, target, *, K: int = 2, target_label=None,
               num_budgets=None, direct_attack=True, structure_attack=True,
               feature_attack=False, disable=False):

        super().attack(target, target_label, num_budgets=num_budgets,
                       direct_attack=direct_attack,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        self.set_normalize(False)
        self.K = K

        target_label = self.target_label.view(-1)
        best_wrong_label = self.strongest_wrong_class(target,
                                                      target_label).view(-1)
        best_wrong_label = best_wrong_label.to(self.device)
        ## Extract the k-hop subgraph + and protential node centered at t    
        subgraph = self.get_subgraph(target, target_label, best_wrong_label)
        ## subgraph 如果子图里有target的已有边，要过滤掉，不能对target产生改变；不允许直接修改目标节点的连接；只能通过修改目标节点的邻居和其他节点来间接影响目标节点
        if not direct_attack:
            condition1 = subgraph.sub_edges[0] != target
            condition2 = subgraph.sub_edges[1] != target
            mask = torch.logical_and(condition1, condition2).float()

        for it in tqdm(range(self.num_budgets), desc='Peturbing graph...',
                       disable=disable):
            ## 是同一个loss求的gradient，诶？但是给的潜在边不一样，那边的潜在边是整个子图，这里的潜在边是top-\Delta条 子图的子图  # 这里的子图是smaller的子图
            non_edge_grad, edge_grad = self.compute_gradients(
                subgraph, target, target_label, best_wrong_label)
            ## 统一表示 调整后的梯度 > 0：执行该操作对攻击有利；调整后的梯度 < 0：执行该操作对攻击不利
            with torch.no_grad():
                edge_grad *= -2 * subgraph.edge_weight + 1
                if not direct_attack:
                    edge_grad *= mask
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1

            max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
            max_non_edge_grad, max_non_edge_idx = torch.max(
                non_edge_grad, dim=0)

            if max_edge_grad > max_non_edge_grad:
                # remove one edge
                subgraph.edge_weight.data[max_edge_idx].fill_(0.)
                u, v = subgraph.sub_edges[:, max_edge_idx].tolist()
                self.remove_edge(u, v, it)
            else:
                # add one edge
                subgraph.non_edge_weight.data[max_non_edge_idx].fill_(1.)
                u, v = subgraph.non_edges[:, max_non_edge_idx].tolist()
                self.add_edge(u, v, it)

        self.set_normalize(True)
        return self

    def compute_gradients(self, subgraph, target, target_label,
                          best_wrong_label):
        #edge_index：non_edges, sub_edges, non_edges.flip(0), sub_edges.flip(0), selfloop.repeat((2, 1))
        edge_weight = torch.cat([   
            subgraph.non_edge_weight, subgraph.edge_weight,     # non_edges, sub_edges
            subgraph.non_edge_weight, subgraph.edge_weight,     # non_edges.flip(0), sub_edges.flip(0)（这个为什么是已有边？无向边？两个方向都得有？）
            subgraph.selfloop_weight
        ], dim=0)       # 边的个数

        row, col = subgraph.edge_index
        norm = (self.degree + 1.).pow(-0.5)
        edge_weight = norm[row] * edge_weight * norm[col]           # 权重归一化
        # SGC模型，self.feat.shape [# nodes of RAW G， dimention of each node's feature] edge_weight subgraph这个子图的边的权重
        # 诶？这里用的全点，部分边，没影响是吗？
        logit = self.surrogate(self.feat, subgraph.edge_index, edge_weight)              # edge_index 定义连接关系；edge_weight 定义连接强度   # logit.shape = [num_nodes of RAW image, # cls] 
        logit = logit[target].view(1, -1) / self.tau        # 调节因子，缓解梯度消失或爆炸
        logit = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(logit, target_label) - \
            F.nll_loss(logit, best_wrong_label)
        return grad(loss,   # 要微分的标量张量（损失函数）
                    [subgraph.non_edge_weight, subgraph.edge_weight],       # 需要计算梯度的变量列表
                    create_graph=False)
        # subgraph = SubGraph(
        #     edge_index=edge_index,              # 包含所有边（正向+反向+自环）
        #     sub_edges=sub_edges,
        #     non_edges=non_edges,
        #     edge_weight=edge_weight,            # 边权重为1
        #     non_edge_weight=non_edge_weight,    # 边权重为0
        #     selfloop_weight=selfloop_weight,    # 边权重为1
        # )
