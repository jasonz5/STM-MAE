from stdmae_arch import MaskST
import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

def main():
    import sys
    from torchsummary import summary
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '2'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = MaskST(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        mask_ratio=0.75,
        encoder_depth=4,
        decoder_depth=1,
        mode="pre-train"
    ).to(device)
    summary(model, (288*3, 307, 1), device=device)


if __name__ == '__main__':
    main()
