import torch
from torch import nn

from .mask import Mask
from .graphwavenet import MLPSIMPLE

'''STDMAE without GWNet, just a vanilla MLP to fetch fetures from short time series'''
class STDMAESIMPLE(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args, with_mae):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        # iniitalize 
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)

        self.backend = MLPSIMPLE(**backend_args)
        self.with_mae = with_mae
        # load pre-trained model
        self.load_pre_trained_model()


    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        checkpoint_dict = torch.load(self.pre_trained_smae_path)
        self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        # reshape
        short_term_history = history_data     # [B, L, N, 1]

        batch_size, _, num_nodes, _ = history_data.shape
        if self.with_mae:
            hidden_states_t = self.tmae(long_history_data[..., [0]]) # B, N, P, d
            hidden_states_s = self.smae(long_history_data[..., [0]]) # B, N, P, d
            hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1) # B, N, P, 2d
        
            # enhance
            out_len=1
            hidden_states = hidden_states[:, :, -out_len, :]  # B, N, 2d
        else:
            hidden_states = None
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1) #[B, 12(P), N, 1]
        
        return y_hat

