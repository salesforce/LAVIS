'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
## FROM: https://github.com/salesforce/ULIP
## TODO: Convert to LAVIS format. Currently only supports functionality for XInstructBLIP

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
from lavis.models.ulip_models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
import torch
import numpy as np
from lavis.common.dist_utils import download_cached_file


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model
        self.num_features = kwargs.embed_dim

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, kwargs.embed_dim ))
        nn.init.normal_(self.pc_projection, std= kwargs.embed_dim  ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text=None, image=None):

        if text is not None:
            text_embed_all = []
            for i in range(text.shape[0]):
                text_for_one_sample = text[i]
                text_embed = self.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

            text_embed_all = torch.stack(text_embed_all)
        else: 
            text_embed_all = None

        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
        else:
            image_embed = None
        
        res = {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()
                }
        return pc_embed


def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model):
    return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']

def ULIP_PointBERT(ulip_v=2):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from lavis.models.ulip_models.pointbert.point_encoder import PointTransformer
    from lavis.models.ulip_models.utils.config import cfg_from_yaml_file
    ## TODO: parse as config
    # config_addr = '/export/home/LAVIS/lavis/models/ulip_models/pointbert/PointTransformer_8192point.yaml'
    url = "https://raw.githubusercontent.com/salesforce/ULIP/48d8d00b1cdb2aee79005817a202816f1c521911/models/pointbert/PointTransformer_8192point.yaml"
    config_addr = download_cached_file(
        url, check_hash=False, progress=True
    )
    config = cfg_from_yaml_file(config_addr)
    pc_feat_dims = 768 
    if ulip_v == "ulip2_scaledup":
        config.model.depth = 18
        transformer_layers = 18
        embed_dim=1280
    else:
        embed_dim=512

        transformer_layers = 12
    point_encoder = PointTransformer(config.model)
    # =====================================================================
    model = ULIP_WITH_IMAGE(embed_dim=embed_dim, vision_width=pc_feat_dims, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=transformer_layers, pc_feat_dims=pc_feat_dims)
                            
    ## TODO: setup config
    if ulip_v == 2:
        cached_file = '/export/share/lxue/shared_models/ULIP-2/objaverse_shapenet_k_5/ULIP-2_pointbert_last.pt'
    elif ulip_v == 1:
        cached_file = '/export/share/lxue/shared_models/ULIP-1/objaverse/ULIP-1_pointbert_last.pt'
    elif ulip_v == 'shapenet':
        cached_file = '/export/share/lxue/shared_models/ULIP-1/objaverse_shapenet/checkpoint_last.pt'
    elif ulip_v == 'objaverse_k_1':
        cached_file = '/export/share/lxue/shared_models/ULIP-2/objaverse_k_1/checkpoint_last.pt'
    elif ulip_v == 'objaverse_shapenet_k_1':
        cached_file = '/export/share/lxue/shared_models/ULIP-2/objaverse_shapenet_k_1/checkpoint_last.pt'
    elif ulip_v == "ulip2_scaledup":
        cached_file = "/export/share/lxue/shared_models/ULIP-2/objaverse_shapenet_k_1_scaled_up/checkpoint_last.pt"
    # url = "https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt"
    # cached_file = download_cached_file(
    #     url, check_hash=False, progress=True
    # )
    ckpt = torch.load(cached_file, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    # model.cuda()
    model.load_state_dict(state_dict, strict=False)
    return model