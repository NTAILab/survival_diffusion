from .base import BaselineBase
import torch
from typing import Any, Literal, List
from ..util import Data, FeatNumber
from copy import copy

class ConditionalLayerNormBlock(torch.nn.Module):
    def __init__(self, in_features: FeatNumber, out_features: int, activation: torch.nn.Module):
        super().__init__()
        linear_feats = in_features.numeric + in_features.categorical
        time_dim = in_features.time
        self.linear = torch.nn.Linear(linear_feats, out_features)
        self.norm = torch.nn.LayerNorm(out_features)
        self.to_gamma_beta = torch.nn.Linear(time_dim, 2 * out_features)
        self.act = activation
        torch.nn.init.zeros_(self.to_gamma_beta.weight)
        torch.nn.init.zeros_(self.to_gamma_beta.bias)
        self.out_feats = FeatNumber(
            numeric=out_features,
            time=time_dim
        )

    def forward(self, data: Data) -> Data:
        gamma, beta = self.to_gamma_beta(data.time_emb).chunk(2, dim=-1)
        if data.X_cat is not None:
            X = torch.cat((data.X_num, data.X_cat), dim=-1)
        else:
            X = data.X_num
        X_tf = self.linear(X)
        norm = self.norm(X_tf)
        result = Data(
            X_num = self.act((1 + gamma) * norm + beta),
            time_emb=data.time_emb
        )
        return result

class LinearWrapper(torch.nn.Linear):
    def __init__(self, in_features: FeatNumber, out_features: int, norm: bool,
                 activation: torch.nn.Module,
                 bias: bool = True, device=None, dtype=None) -> None:
        input_feat_n = in_features.numeric + in_features.categorical + in_features.time
        self.f_use_time = in_features.time > 0
        super().__init__(input_feat_n, out_features, bias, device, dtype)
        self.norm = torch.nn.LayerNorm(out_features) if norm else torch.nn.Identity()
        self.act = activation
        self.out_feats = FeatNumber(
            numeric=out_features
        )
    
    def forward(self, data: Data) -> Data:
        if self.f_use_time:
            input = torch.cat((data.time_emb, data.X_num), dim=-1)
        else:
            input = data.X_num
        if data.X_cat is not None:
            input = torch.cat((input, data.X_cat), dim=-1)
        result = Data(
            X_num=self.act(self.norm(super().forward(input)))
        )
        return result

class Dropout(torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
    
    def forward(self, input: Data) -> Data:
        assert input.X_cat is None
        result = Data(
            X_num=super().forward(input.X_num),
            X_cat=None,
            time_emb=input.time_emb
        )
        return result

class MLPBaseline(BaselineBase):
    
    def _get_dropout_idx(self, layer_n: int, dropout_p: float) -> List[int]:
        if dropout_p <= 0:
            return []
        if layer_n <= 3:
            return [0]
        return [0, layer_n - 2]
    
    def __init__(self,
                 layer_n: int,
                 input_dim: FeatNumber,
                 hidden_dim: int,
                 activation: str,
                 normalization: Literal['none', 'layer', 'conditional'],
                 dropout: float,
                 seed: int | None = None
                 ):
        super().__init__(output_dim=2, seed=seed)
        layers = []
        inp_d = copy(input_dim)
        act = getattr(torch.nn, activation)
        match normalization:
            case 'layer':
                get_layer = lambda inp_d, out_d: LinearWrapper(
                    in_features=inp_d,
                    out_features=out_d,
                    activation=act(),
                    norm=True
                )
            case 'none':
                get_layer = lambda inp_d, out_d: LinearWrapper(
                    in_features=inp_d,
                    out_features=out_d,
                    activation=act(),
                    norm=False
                )
            case 'conditional':
                get_layer = lambda inp_d, out_d: ConditionalLayerNormBlock(
                    in_features=inp_d,
                    out_features=out_d,
                    activation=act()
                )
            case _:
                raise ValueError("Unknown normalization type")
        dropout_idx = self._get_dropout_idx(layer_n, dropout)
        for i in range(layer_n - 1):
            out_d = hidden_dim
            layers.append(get_layer(inp_d, out_d))
            self.register_module(f'linear_{i}', layers[-1])
            inp_d = layers[-1].out_feats
            if i in dropout_idx:
                layers.append(Dropout(dropout))
                self.register_module(f'dropout_{i}', layers[-1])
        layers.append(LinearWrapper(
            in_features=inp_d,
            out_features=2,
            norm=False,
            activation=torch.nn.Identity()
        ))
        self.register_module(f'linear_output', layers[-1])
        self.layers = layers
        
    def forward(self, data: Data) -> torch.Tensor:
        X = data
        for layer in self.layers:
            X = layer(X)
        return X.X_num
