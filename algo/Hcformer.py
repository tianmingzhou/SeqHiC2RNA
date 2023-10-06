import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange
from einops.layers.torch import Rearrange

from transformers import PreTrainedModel

from algo.module import Residual, AttentionPool, Attention, TargetLengthCrop, GELU
from algo.module import ConvBlock, exponential_linspace_int, map_values, exists,\
poisson_loss, fetch_pred

from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot

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
        half_seq_dim = config.seq_dim // 2
    
        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_seq_dim, 15, padding = 7, stride=2),
            Residual(ConvBlock(half_seq_dim)),
            AttentionPool(half_seq_dim, pool_size = 8)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_seq_dim, config.seq_dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_seq_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.dim // config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = config.dim // config.heads
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

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

        # create trunk sequential module

        self.seq_conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),        
        )

        self._trunk = nn.Sequential(
            self.transformer,
            self.crop_final,
            self.final_pointwise
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
    def trunk(self):
        return self._trunk

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
        target = None,
        index = None,
        hic_1d = None,
        return_fetch_pred = None,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None
    ):
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        if exists(target_length):
            self.set_target_length(target_length)

        # trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = self.seq_conv(x)
        x = torch.concat((x, hic_1d.unsqueeze(-1)), dim=2)
        x = self._trunk(x)

        if no_batch:
            x = rearrange(x, '() ... -> ...')

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f'head {head} not found'
            out = out[head]

        if exists(target):
            assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

            return poisson_loss(out, target, index)

        if return_embeddings:
            return out, x

        if return_fetch_pred:
            return fetch_pred(out, index)

        return out
