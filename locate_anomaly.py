import sys
sys.path.append("./Dinomaly")

import torch
import torch.nn as nn
import os

from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp,LinearAttention2
from dataset import MVTecDataset
from utils import visualize_one
from functools import partial
import warnings

warnings.filterwarnings("ignore")

def locate(image_path,iter,save_name,save_dir ='./Dinomaly/saved_results'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder_name = 'dinov2reg_vit_small_14'
    # encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    state_dict = torch.load(os.path.join(save_dir, save_name, f'model_{encoder_name}_{iter}.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    return visualize_one(model,image_path,iter,device)