import torch
import torch.nn as nn
from typing import List, Callable, Optional


class CVAE(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, task_dim: int, latent_dim: int = 32,
                 encoder_hidden: List[int] = [128, 64], prior_hidden: List[int] = [128, 64], decoder_hidden: List[int] = [128, 64]):
        super().__init__()
        
        self._encoder = MLP([observation_dim + action_dim + task_dim] + encoder_hidden + [latent_dim * 2])
        self._prior = MLP([observation_dim + task_dim] + prior_hidden + [latent_dim * 2])
        self._decoder = MLP([latent_dim + observation_dim + task_dim] + decoder_hidden + [action_dim * 2])

    def sample(self, mu_logvar: torch.tensor):
        mu = mu_logvar[:,:mu_logvar.shape[-1] // 2]
        std = (mu_logvar[:,mu_logvar.shape[-1] // 2:] / 2).exp()
        return torch.empty_like(mu).normal_() * std + mu

    def encode(self, obs: torch.tensor, action: torch.tensor, task: torch.tensor = None, sample: bool = False):
        if task is not None:
            mu_logvar = self._encoder(torch.cat((obs, action, task), -1))
        else:
            mu_logvar = self._encoder(torch.cat((obs, action), -1))
        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def prior(self, obs: torch.tensor, task: torch.tensor = None, sample: bool = False):
        if task is not None:
            mu_logvar = self._prior(torch.cat((obs, task), -1))
        else:
            mu_logvar = self._prior(torch.cat((obs,), -1))

        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def decode(self, latent: torch.tensor, obs: torch.tensor, task: torch.tensor = None, sample: bool = False):
        if task is not None:
            mu_logvar = self._decoder(torch.cat((latent, obs, task), -1))
        else:
            mu_logvar = self._decoder(torch.cat((latent, obs), -1))
        if sample:
            return mu_logvar, self.sample(mu_logvar)
        else:
            return mu_logvar

    def forward(self, obs: torch.tensor, task: torch.tensor = None):
        z = self.prior(obs, task, sample=True)[1]
        mu_logvar = self.decode(z, obs, task)
        return mu_logvar[:,:mu_logvar.shape[-1] // 2], (mu_logvar[:,mu_logvar.shape[-1] // 2:] / 2).exp()

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
    def __init__(self, layer_widths: List[int], final_activation: Callable = lambda x: x, bias_linear: bool = False, extra_head_layers: List[int] = None):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError('Layer widths needs at least an in-dimension and out-dimension')

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        self._head = extra_head_layers is not None

        linear = BiasLinear if bias_linear else nn.Linear

        for idx in range(len(layer_widths) - 1):
            self.seq.add_module(f'fc_{idx}', linear(layer_widths[idx], layer_widths[idx + 1]))
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f'relu_{idx}', nn.ReLU())

        if extra_head_layers is not None:
            self.pre_seq = self.seq[:-2]
            self.post_seq = self.seq[-2:]

            self.head_seq = nn.Sequential()
            extra_head_layers = [layer_widths[-2] + layer_widths[-1]] + extra_head_layers

            for idx, (infc, outfc) in enumerate(zip(extra_head_layers[:-1], extra_head_layers[1:])):
                self.head_seq.add_module(f'fc_{idx}', linear(extra_head_layers[idx], extra_head_layers[idx + 1]))
                if idx < len(extra_head_layers) - 2:
                    self.seq.add_module(f'relu_{idx}', nn.ReLU())
                
    def forward(self, x: torch.tensor, acts: Optional[torch.tensor] = None):
        if self._head and acts is not None:
            h = self.pre_seq(x)
            head_input = torch.cat((h,acts), -1)
            return self._final_activation(self.post_seq(h)), self.head_seq(head_input)
        else:
            return self._final_activation(self.seq(x))
        

if __name__ == '__main__':
    mlp = MLP([1,5,8,2])
    x = torch.empty(10,1).normal_()
    print(mlp(x).shape)
