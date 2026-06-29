"""CytoDiff / CytoMal architecture (inference build)."""
import torch
import torch.nn as nn


class StainAwareInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, 1)
        stain = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        with torch.no_grad():
            self.proj.weight.copy_((torch.eye(3) * 0.5 + stain * 0.5).view(3, 3, 1, 1))
            self.proj.bias.zero_()
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        return x + self.gamma * self.proj(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // r, 8)), nn.ReLU(True),
                                nn.Linear(max(ch // r, 8), ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)


class TransformerBranch(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=192, depth=2, heads=3):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, patch)
        n = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.size(0)
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self.pos
        return self.norm(self.encoder(t)[:, 0])


class CBAF(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, a, b):
        t = torch.stack([a, b], 1)
        at, _ = self.attn(t, t, t)
        return self.norm(t + at).flatten(1)


class CytoDiff(nn.Module):
    def __init__(self, num_classes, cnn="efficientnet_b0", dim=192, pretrained=False, img_size=224):
        super().__init__()
        import timm
        self.saim = StainAwareInput()
        self.cnn = timm.create_model(cnn, features_only=True, pretrained=pretrained, in_chans=3)
        ch = self.cnn.feature_info.channels()[-1]
        self.se = SEBlock(ch)
        self.proj = nn.Linear(ch, dim)
        self.vit = TransformerBranch(img_size, dim=dim)
        self.fuse = CBAF(dim)
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Dropout(0.2), nn.Linear(2 * dim, num_classes))

    def forward(self, x):
        x = self.saim(x)
        f = self.se(self.cnn(x)[-1])
        return self.head(self.fuse(self.proj(f.mean(dim=(2, 3))), self.vit(x)))


def build_cytodiff(num_classes, cnn="efficientnet_b0", pretrained=False, img_size=224):
    return CytoDiff(num_classes, cnn=cnn, pretrained=pretrained, img_size=img_size)
