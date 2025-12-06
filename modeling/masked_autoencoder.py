import torch
import torch.nn as nn

from dataclasses import dataclass, field

from modeling import TransformerLayer


@dataclass
class MaskedAutoencoderConfig():
    image_size: tuple = field(
        default=(40, 30), metadata={"help": "number of decoder heads"}
    )
    patch_size: tuple = field(
        default=(4, 3), metadata={"help": "number of decoder heads"}
    )
    mask_ratio: float = field(
        default=0.5, metadata={"help": "probability of masking"}
    )
    n_channels: int = field(
        default= 6, metadata={"help": "number of decoder heads"}
    )

    decoder_heads: int = field(
        default=8, metadata={"help": "number of decoder heads"}
    )
    decoder_layers: int = field(
        default=1, metadata={"help": "numebr of decoder layers"}
    )
    decoder_dim_per_head: int = field(
        default=64, metadata={"help": "dim of decoder per heads"}
    )

    encoder_rate: int = field(
        default=4, metadata={"help": "ffn expansion"}
    )
    encoder_layers: int = field(
        default=10, metadata={"help": "number of encoder layers"}
    )
    encoder_heads: int = field(
        default=8, metadata={"help": "number of encoder heads"}
    )
    encoder_dim_per_head: int = field(
        default=64, metadata={"help": "dim of encoder per heads"}
    )


    dropout: int = field(
        default=0.1, metadata={"help": "dropout rate"}
    )

class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        cfg = MaskedAutoencoderConfig(),
    ):
        super().__init__()

        self.h, self.w = cfg.image_size
        self.patch_h, self.patch_w = cfg.patch_size

        self.num_patches = (self.h // self.patch_h) * (self.w // self.patch_w)

        self.num_mask = int(cfg.mask_ratio * self.num_patches)
        self.in_channels = cfg.n_channels

        self.patch_embed = nn.Linear(self.patch_h * self.patch_w * cfg.n_channels, cfg.encoder_dim_per_head * cfg.encoder_heads)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, cfg.encoder_dim_per_head * cfg.encoder_heads))

        self.adapter = nn.Linear(cfg.encoder_dim_per_head * cfg.encoder_heads, cfg.decoder_heads * cfg.decoder_dim_per_head)

        self.mask_ratio = cfg.mask_ratio
        self.mask_embed = nn.Parameter(torch.randn(cfg.decoder_dim_per_head * cfg.decoder_heads))

        self.encoder = nn.ModuleList(
            [TransformerLayer(cfg.encoder_heads, cfg.encoder_heads * cfg.encoder_dim_per_head, cfg.encoder_rate, cfg.dropout) for _ in range(cfg.encoder_layers)]
        )

        self.decoder = nn.ModuleList(
            [TransformerLayer(cfg.decoder_heads, cfg.decoder_heads * cfg.decoder_dim_per_head, cfg.encoder_rate, cfg.dropout) for _ in range(cfg.decoder_layers)]
        )
        self.decoder_pos_embed = nn.Embedding(self.num_patches, cfg.decoder_dim_per_head * cfg.decoder_heads)

        self.head = nn.Linear(cfg.decoder_dim_per_head * cfg.decoder_heads, self.patch_h * self.patch_w * cfg.n_channels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        b = x.size(0)

        # patch partition
        patches = x.view(
            b, self.in_channels,
            self.h // self.patch_h, self.patch_h,
            self.w // self.patch_w, self.patch_w,
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)                                                          # (B, num_patches, patch_pixel)

        # masking
        shuffle_indices = torch.rand(b, self.num_patches, device=x.device).argsort()                                          # (B, num_mask)
        masked_indices, unmasked_indices = shuffle_indices[:, :self.num_mask], shuffle_indices[:, self.num_mask:]
        batch_indice = torch.arange(b, device=x.device).unsqueeze(-1)
        masked_patches, unmasked_patches = patches[batch_indice, masked_indices], patches[batch_indice, unmasked_indices]     # (B, ..., patch_pixel)

        # forward unmasked patches
        unmasked_tokens = self.patch_embed(unmasked_patches)                                                                  # (B, num_unmask, dim)
        unmasked_tokens += self.pos_embed.repeat(b, 1, 1)[batch_indice, unmasked_indices]                                     # (B, num_unmask, dim)
        for layer in self.encoder:
            unmasked_tokens = layer(unmasked_tokens)
        unmasked_tokens = self.adapter(unmasked_tokens)                                                                       # (B, num_unmask, dim)

        # process masked patches
        mask_tokens = self.mask_embed[None, None, :].repeat(b, self.num_mask, 1)                                              # (B, num_mask, dim)
        mask_tokens += self.decoder_pos_embed(masked_indices)                                                                 # (B, num_mask, dim)

        # merge masked and unmasked tokens
        concat_tokens = torch.cat([mask_tokens, unmasked_tokens], dim=1)                                                      # (B, num_patches, dim)
        out_tokens = torch.empty_like(concat_tokens, device=x.device)                                                         # (B, num_patches, dim)
        out_tokens[batch_indice, shuffle_indices] = concat_tokens                                                             # (B, num_patches, dim)

        for layer in self.decoder:
            out_tokens = layer(out_tokens)                                                                                    # (B, num_patches, dim)

        decoded_masked_tokens = out_tokens[batch_indice, masked_indices]                                                      # (B, num_mask, dim)
        pred_mask_pixels = self.head(decoded_masked_tokens)                                                                   # (B, num_mask, patch_pixel)

        results = {"target": masked_patches, "pred": pred_mask_pixels}
        return results