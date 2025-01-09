import torch.nn as nn
from functorch.einops import rearrange


class TimeEmbedding(nn.Module):

    def __init__(self, dim_encoded_time=None, dim_input=None, trainable=True):
        super().__init__()
        if trainable:
            if dim_input is None or dim_encoded_time is None:
                raise RuntimeError("'dim_input' and 'dim_encoded_time' must be specficed when trainable is True")
            else:
                self.linear1 = nn.Linear(dim_encoded_time, dim_input)
                self.activation = nn.GELU()
                self.linear2 = nn.Linear(dim_input, dim_input)
        else:
            self.linear1 = nn.Identity()
            self.activation = nn.Identity()
            self.linear2 = nn.Identity()

    def forward(self, x, t):
        t = self.linear2(self.activation(self.linear1(t)))
        if len(x.shape) == 4:
            t = rearrange(t, "b c -> b c 1 1")
        elif len(x.shape) == 3:
            t = rearrange(t, "b c -> b 1 c")
        return x + t
