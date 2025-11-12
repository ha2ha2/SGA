from greatx.nn.models import GCN
from greatx.training import Trainer
from torch_geometric.datasets import Planetoid
# Any PyG dataset is available!
dataset = Planetoid(root='.', name='Cora')
data = dataset[0]
model = GCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device='cuda:0') # or 'cpu'
trainer.fit(data, mask=data.train_mask)
trainer.evaluate(data, mask=data.val_mask)


from greatx.attack.targeted import RandomAttack
attacker = RandomAttack(data)

attacker.reset()
attacker.attack(1, num_budgets=3) # attacking target node `1` with `3` edges
attacked_data = attacker.data()
edge_flips = attacker.edge_flips()


from greatx.attack.untargeted import RandomAttack
attacker = RandomAttack(data)

attacker.reset()
attacker.attack(num_budgets=0.05) # attacking the graph with 5% edges perturbations
attacked_data = attacker.data()
edge_flips = attacker.edge_flips()

print()
