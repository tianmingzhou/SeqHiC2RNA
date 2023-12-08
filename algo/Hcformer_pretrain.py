import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import torchvision.transforms as transforms
import sys
sys.path.append('..')

from einops import rearrange
from einops.layers.torch import Rearrange

from transformers import PreTrainedModel

from algo.module import Residual, AttentionPool, Attention, Hic_Attention, TargetLengthCrop, GELU
from algo.module import ConvBlock, map_values, exists

from algo.config import HcformerConfig

class Hcformer(PreTrainedModel):
    config_class = HcformerConfig
    base_model_prefix = 'hcformer'

    @staticmethod
    def from_hparams(**kwargs):
        return Hcformer(HcformerConfig(**kwargs))
    
    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        twice_dim = config.dim * 2
        self.seq_dim = config.seq_dim
        self.hic_1d = config.hic_1d
        self.hic_2d = config.hic_2d
        self.depth = config.depth

        self.dim_transform = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, config.dim)
        )

        if config.hic_1d:
            # hic_1d data transformation
            self.hic_1d_transform = nn.Sequential(
                nn.Linear(config.hic_1d_feat_num, config.hic_1d_feat_dim),
                nn.LayerNorm(config.hic_1d_feat_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hic_1d_feat_dim, config.hic_1d_feat_dim)
            )

        if config.hic_2d:
            self.Gaussianblur = transforms.GaussianBlur(kernel_size=5, sigma=1.5)
            self.hic_2d_CNN = nn.ModuleList()
            for i in range(config.depth):
                self.hic_2d_CNN.append(nn.Sequential(
                    nn.Conv2d(2, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(64, config.heads, kernel_size=3, padding=1),
                ))

        # transformer
        self.attention_block = nn.ModuleList()
        self.MLP_block = nn.ModuleList()
        for i in range(config.depth):
            self.attention_block.append(Residual(
                    Hic_Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.dim // config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = config.dim // config.heads,
                        hic_2d = config.hic_2d,
                        post_dropout=config.dropout_rate,
                    )))
            self.MLP_block.append(Residual(
                nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                )))

        # final pooling
        self.pool = nn.Sequential(
            Rearrange('b n d -> b d n'),
            AttentionPool(config.dim, pool_size=8),
            Rearrange('b d n -> b n d')
        )

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

       # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(config.dim, twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def heads(self):
        return self._heads
    
    # Haven't finished using this part
    # def trunk_checkpointed(self, x):
    #     x = rearrange(x, 'b n d -> b d n')
    #     x = self.stem(x)
    #     x = self.conv_tower(x)
    #     x = rearrange(x, 'b d n -> b n d')
    #     x = checkpoint_sequential(self.transformer, len(self.transformer), x)
    #     x = self.crop_final(x)
    #     x = self.final_pointwise(x)
    #     return x

    def forward(
        self,
        x,
        hic_1d = None,
        hic_2d = None,
        head = None,
    ):
        
        if exists(self.dim_transform):
            x = self.dim_transform(x)
    
        x = self.pool(x)

        if self.hic_2d:
            x_oprod = rearrange(torch.matmul(x, x.transpose(1, 2)).unsqueeze(-1), 'b h w c -> b c h w') / 256
            hic_2d = self.Gaussianblur(rearrange(hic_2d.unsqueeze(-1), 'b h w c -> b c h w'))
            hic_2d = torch.concat((hic_2d, x_oprod), dim=1)
        if self.hic_1d:
            hic_1d = self.hic_1d_transform(hic_1d)
            x = x + hic_1d

        for i in range(self.depth):
            if self.hic_2d:
                hic_attn = self.hic_2d_CNN[i](hic_2d)
                x = self.attention_block[i](x, hic_2d=hic_attn)
                x = self.MLP_block[i](x)
            else:
                x = self.attention_block[i](x)
                x = self.MLP_block[i](x)

        x = self.crop_final(x)
        x = self.final_pointwise(x)

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f'head {head} not found'
            out = out[head]

        return out
