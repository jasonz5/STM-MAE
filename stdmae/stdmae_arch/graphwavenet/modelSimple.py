import torch
from torch import nn
import torch.nn.functional as F

class MLPSIMPLE(nn.Module):
    def __init__(self, num_nodes, in_dim=2, time_dim=12, out_dim=12,skip_channels=256,end_channels=512, **kwargs):

        super(MLPSIMPLE, self).__init__()
        self.in_dim = in_dim
        self.time_dim = time_dim
        self.start_mlp = nn.Sequential(nn.Linear(in_dim*time_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

    def forward(self, input, hidden_states):
        """
        input [B, L, N, 1]
        hidden_states [B, N, 2d]
        """
        # reshape input: [B, L, N, C] -> [B, N, C, L]
        input = input.permute(0, 2, 3, 1)
        input = input[:, :, :self.in_dim, :] # [B, N, 2, L]
        batch_size, num_nodes, _, _ = input.shape
        input = input.reshape(batch_size,num_nodes, 24)

        skip = self.start_mlp(input) # B, N, 256(D)
        if hidden_states:
            hidden_states_t = self.fc_his_t(hidden_states[:,:,:96])        # B, N, 256(D)
            hidden_states_s = self.fc_his_s(hidden_states[:,:,96:])        # B, N, 256(D)
            skip = skip + hidden_states_t
            skip = skip + hidden_states_s
        skip = skip.transpose(1, 2).unsqueeze(-1) # B, 256(D), N, 1
        # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1) # B, 256(D), N, 1
        # hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1) # B, D, N, 1

        
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) # B, 12(D), N, 1

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        return x