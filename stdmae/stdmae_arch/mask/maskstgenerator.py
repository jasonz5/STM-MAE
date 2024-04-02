import random
from torch import nn

#BUG: MaskSTGenerator输出在temporal和spatial俩个indices，不知道后面怎么用在encoder、decoder中。
class MaskSTGenerator(nn.Module):
    """Mask generator for spacetime-agnostic random masking."""

    def __init__(self, num_spatial_tokens, num_temporal_tokens, mask_ratio):
        super().__init__()
        self.num_spatial_tokens = num_spatial_tokens
        self.num_temporal_tokens = num_temporal_tokens
        self.total_tokens = num_spatial_tokens * num_temporal_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        # Create a list of all possible token positions in space-time grid
        all_tokens = list(range(int(self.total_tokens)))
        random.shuffle(all_tokens)

        # Calculate the number of tokens to mask
        mask_len = int(self.total_tokens * self.mask_ratio)

        # Split the tokens into masked and unmasked
        self.masked_tokens = all_tokens[:mask_len]
        self.unmasked_tokens = all_tokens[mask_len:]

        if self.sort:
            self.masked_tokens.sort()
            self.unmasked_tokens.sort()

        # Convert linear indices to spatial and temporal indices
        self.masked_spatial_indices = [idx % self.num_spatial_tokens for idx in self.masked_tokens]
        self.masked_temporal_indices = [idx // self.num_spatial_tokens for idx in self.masked_tokens]

        self.unmasked_spatial_indices = [idx % self.num_spatial_tokens for idx in self.unmasked_tokens]
        self.unmasked_temporal_indices = [idx // self.num_spatial_tokens for idx in self.unmasked_tokens]

        return self.unmasked_spatial_indices, self.unmasked_temporal_indices, self.masked_spatial_indices, self.masked_temporal_indices

    def forward(self):
        return self.uniform_rand()
