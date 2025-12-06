import torch
import torch.nn as nn

from modeling import TransformerLayer


class PatchLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (B, C * L, H, W)
    ---------------------------------------------------------------------------
    output          float           (B, N, E)
    ===========================================================================
    """
    def __init__(
        self,
        num_channels,
        embed_dim,
        image_size,
        patch_size,
        dropout = 0.0
    ):
        super().__init__()

        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                           # (B, E, IH/P, IW/P)
        x = x.reshape(B, x.shape[1], -1)            # (B, E, IH/P * IW/P) --> (B, E, N)
        x = x.permute(0, 2, 1)                      # (B, N, E)
        x  = x + self.pos_embedding                 # (B, N, E)
        x = self.dropout(x)
        return x
    
class ViT(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (B, C * L, H, W)
    ---------------------------------------------------------------------------
    output          float           (B, N, E)
    ===========================================================================
    """
    def __init__(
        self,
        n_channels,
        embed_dim,
        n_layers,
        n_attention_heads,
        rate,
        image_size,
        patch_size,
        dropout=0.1
    ):
        self.embedding = PatchLayer(n_channels, embed_dim, image_size, patch_size, dropout)
        self.encoder    = nn.ModuleList(
            [TransformerLayer(n_attention_heads, embed_dim, rate, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm()
    
    def forward(self, x):
        x = self.embedding(x)                       # (B, N, E)
        for layer in self.encoder:
            x = layer(x)                            # (B, N, E)
        x = self.ln(x)                              # (B, N, E)
        return x