import numpy as np
import torch
import torch.nn as nn


class Wide(nn.Module):
    def __init__(self, wide_dim, output_dim=1):
        super(Wide, self).__init__()
        self.wlinear = nn.Linear(wide_dim, output_dim)

    def forward(self, X):
        out = self.wlinear(X)
        return out


def dense_layer(inp, out, p, bn=False):
    layers = [nn.Linear(inp, out), nn.LeakyReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm1d(out))
    layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


class Deep(nn.Module):
    def __init__(self, hidden_layers, embed_input, continuous_cols, deep_column_idx, batchnorm = False,
                 dropout = None, embed_p = 0.0, output_dim=1):

        super(Deep, self).__init__()
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        # Embeddings
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_" + col: nn.Embedding(val, dim)
                for col, val, dim in self.embed_input
            }
        )
        self.embed_dropout = nn.Dropout(embed_p)
        emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])

        # Continuous
        cont_inp_dim = len(self.continuous_cols)

        # Dense Layers
        input_dim = emb_inp_dim + cont_inp_dim
        hidden_layers = [input_dim] + hidden_layers
        if not dropout:
            dropout = [0.0] * len(hidden_layers)
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    hidden_layers[i - 1], hidden_layers[i], dropout[i - 1], batchnorm
                ),
            )

        self.out = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, X):

        # Embeddings
        x = [
            self.embed_layers["emb_layer_" + col](
                X[:, self.deep_column_idx[col]].long()
            )
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(x, 1)
        x = self.embed_dropout(x)

        # Continuous
        cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
        x_cont = X[:, cont_idx].float()
        x = torch.cat([x, x_cont], 1)

        # Dense -> out
        return self.out(self.dense(x))