import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output

class TransformerLayersST(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):  #shape: B, N*P*(1-r), d
        B, NL, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        # src = src.view(B, NL, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1)#.view(B, NL, D)
        return output
    