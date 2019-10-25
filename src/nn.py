import torch
import torch.nn as nn
from typing import List, Callable, Optional


class BiasLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_size: Optional[int] = None):
        super().__init__()
        if bias_size is None:
            bias_size = out_features

        self._linear = nn.Linear(in_features, out_features)
        self._bias = nn.Parameter(torch.empty_like(self._linear.bias).normal_(0, 1. / bias_size))
        self._weight = nn.Parameter(torch.empty(bias_size, out_features))
        nn.init.xavier_normal_(self._weight)

    def forward(self, x: torch.tensor):
        return self._linear(x) + self._bias @ self._weight


class MLP(nn.Module):
    def __init__(self, layer_widths: List[int], final_activation: Callable = lambda x: x, bias_linear: bool = False):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError('Layer widths needs at least an in-dimension and out-dimension')

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        linear = BiasLinear if bias_linear else nn.Linear
        for idx in range(len(layer_widths) - 1):
            self.seq.add_module(f'fc_{idx}', linear(layer_widths[idx], layer_widths[idx + 1]))
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f'relu_{idx}', nn.ReLU())

    def forward(self, x: torch.tensor):
        return self._final_activation(self.seq(x))
        

if __name__ == '__main__':
    mlp = MLP([1,5,8,2])
    x = torch.empty(10,1).normal_()
    print(mlp(x).shape)
