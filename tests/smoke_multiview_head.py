"""Small CPU forward-pass check for the proposed prediction head."""

import torch
from torch_geometric.data import HeteroData

import fraudGT  # noqa: F401 - runs custom registrations
from fraudGT.graphgym.config import cfg, set_cfg
from fraudGT.graphgym.register import head_dict, loss_dict


class TinyDataset:
    def __init__(self):
        self.splits = {name: self._graph() for name in ('train', 'val', 'test')}

    @staticmethod
    def _graph():
        graph = HeteroData()
        graph['node'].x = torch.zeros(4, 8)
        store = graph['node', 'to', 'node']
        store.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        store.edge_attr = torch.randn(3, 5)
        store.y = torch.tensor([0, 1, 0])
        store.split_mask = torch.ones(3, dtype=torch.bool)
        return graph

    def __getitem__(self, key):
        return self.splits['train'] if key == 0 else self.splits[key]


def main():
    set_cfg(cfg)
    cfg.device = 'cpu'
    cfg.dataset.task_entity = ('node', 'to', 'node')
    cfg.model.loss_fun = 'class_weighted_focal'
    cfg.model.loss_fun_weight = [1.0, 6.0]

    dataset = TinyDataset()
    head = head_dict['multiview_hetero_edge'](8, 1, dataset)
    batch = dataset['train'].clone()
    store = batch[cfg.dataset.task_entity]
    store.edge_attr = torch.randn(3, 8)
    store.raw_edge_attr = torch.randn(3, 5)
    store.e_id = torch.arange(3)
    store.input_id = torch.arange(3)
    batch.split = 'train'

    logits, labels = head(batch)
    loss, probabilities = loss_dict['class_weighted_focal'](logits.squeeze(-1), labels)

    assert logits.shape == (3, 1)
    assert labels.shape == (3,)
    assert probabilities.shape == (3,)
    assert torch.isfinite(loss)
    assert torch.all((head.last_gate >= 0) & (head.last_gate <= 1))
    print('MV-IA-FraudGT smoke test: OK')
    print('gate alpha:', head.last_gate.squeeze(-1).tolist())


if __name__ == '__main__':
    main()
