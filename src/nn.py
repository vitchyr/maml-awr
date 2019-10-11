import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    def __init__(self, layer_widths: List[int]):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError('Layer widths needs at least an in-dimension and out-dimension')

        self.seq = nn.Sequential()
        for idx in range(len(layer_widths) - 1):
            self.seq.add_module(f'fc_{idx}', nn.Linear(layer_widths[idx], layer_widths[idx + 1]))
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f'relu_{idx}', nn.ReLU())

    def forward(self, x: torch.tensor):
        return self.seq(x)


if __name__ == '__main__':
    mlp = MLP([1,5,8,2])
    x = torch.empty(10,1).normal_()
    print(mlp(x).shape)
