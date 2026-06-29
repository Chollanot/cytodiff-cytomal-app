"""
CytoDiff — the novel single-cell classifier for WBC differential.

Pipeline 1 scope: Olympus CX33 microscope images, annotated as COCO/YOLOv8.
(Single imaging domain — smartphone fusion is deferred to a later pipeline.)

Architecture (distinct from a plain ResNet / EfficientNet / ViT):

    input RGB
       │
   [SAIM] Stain-Aware Input Module  (learnable, Wright-Giemsa initialised)
       │  enhanced image
       ├───────────────► [Local branch]  pretrained EfficientNet-B0 CNN ─► SE attention ─► GAP ─► cnn_vec
       │
       └───────────────► [Global branch] small ViT over the same image ─────────────────► cls ─► tr_vec
                                                   │
                                  [CBAF] Cross-Branch Attention Fusion (multi-head attn
                                          between cnn_vec and tr_vec, not concat)
                                                   │
                                              classifier head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
class StainAwareInput(nn.Module):
    """Learnable colour transform that boosts stain separation before the
    encoders. Initialised toward nucleus (purple/blue) vs cytoplasm (pink)
    directions, then refined during training. On CX33 data this also absorbs
    illumination / white-balance variation between fields and slides."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        # init near identity + a mild Wright-Giemsa stain matrix
        stain = torch.tensor([
            [0.65, 0.70, 0.29],   # nucleus-ish
            [0.07, 0.99, 0.11],   # cytoplasm-ish
            [0.27, 0.57, 0.78],   # background-ish
        ], dtype=torch.float32)
        with torch.no_grad():
            self.proj.weight.copy_((torch.eye(3) * 0.5 + stain * 0.5).view(3, 3, 1, 1))
            self.proj.bias.zero_()
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        return x + self.gamma * self.proj(x)


# ----------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention on the CNN feature map."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


# ----------------------------------------------------------------------
class TransformerBranch(nn.Module):
    """A small Vision-Transformer for global nuclear/cytoplasmic context."""
    def __init__(self, img_size=224, patch=16, dim=192, depth=2, heads=3):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_tokens = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        b = x.size(0)
        t = self.patch(x).flatten(2).transpose(1, 2)      # (B, N, dim)
        cls = self.cls.expand(b, -1, -1)
        t = torch.cat([cls, t], dim=1) + self.pos
        t = self.encoder(t)
        return self.norm(t[:, 0])                          # (B, dim) CLS token


# ----------------------------------------------------------------------
class CrossBranchAttentionFusion(nn.Module):
    """Fuse the CNN token and the Transformer token with multi-head attention
    so each branch can re-weight the other (instead of plain concatenation)."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cnn_vec, tr_vec):
        tokens = torch.stack([cnn_vec, tr_vec], dim=1)     # (B, 2, dim)
        attended, _ = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attended)
        return fused.flatten(1)                             # (B, 2*dim)


# ----------------------------------------------------------------------
class CytoDiff(nn.Module):
    def __init__(self, num_classes, cnn_name="efficientnet_b0",
                 dim=192, pretrained=True, img_size=224):
        super().__init__()
        import timm
        self.saim = StainAwareInput()

        # local morphology branch (pretrained CNN, last feature map only).
        # No explicit out_indices -> robust across timm versions/backbones;
        # we simply take the last returned feature map in forward().
        self.cnn = timm.create_model(
            cnn_name, features_only=True, pretrained=pretrained, in_chans=3)
        cnn_ch = self.cnn.feature_info.channels()[-1]
        self.se = SEBlock(cnn_ch)
        self.cnn_proj = nn.Linear(cnn_ch, dim)

        # global context branch
        self.vit = TransformerBranch(img_size=img_size, dim=dim)

        # fusion + head
        self.fusion = CrossBranchAttentionFusion(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Dropout(0.2),
            nn.Linear(2 * dim, num_classes))

    def forward(self, x):
        x = self.saim(x)
        f = self.cnn(x)[-1]                 # (B, C, h, w)
        f = self.se(f)
        cnn_vec = self.cnn_proj(f.mean(dim=(2, 3)))   # (B, dim)
        tr_vec = self.vit(x)                          # (B, dim)
        fused = self.fusion(cnn_vec, tr_vec)
        return self.head(fused)


def build_cytodiff(num_classes, cfg=None, pretrained=True):
    cnn_name = "efficientnet_b0"
    img_size = 224
    if cfg is not None:
        cnn_name = cfg.get("models", {}).get("cnn_branch", cnn_name)
        img_size = cfg.get("train", {}).get("img_size", img_size)
    return CytoDiff(num_classes, cnn_name=cnn_name,
                    pretrained=pretrained, img_size=img_size)


if __name__ == "__main__":
    m = build_cytodiff(5, pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    print("output:", out.shape)  # expect (2, 5)
