import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Patch Embedding layer that splits the image into overlapping patches and embeds them.
    """
    def __init__(self, img_size=224, patch_size=16, stride=8, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        #self.n_patches = ((img_size - patch_size) // stride + 1) ** 2
        self.padding = (patch_size//2)
        #self.n_patches = ((img_size+self.padding) // stride) ** 2
        self.n_patches = int(((img_size + 2*self.padding - self.patch_size)/stride+1)**2)
        #print()
        #print()
        #print(self.n_patches, ((img_size + 2*self.padding - self.patch_size)/stride+1)**2)
        #print()
        #print()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size // 2))

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """
    Attention layer used in the Vision Transformer.
    """
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block used in the Vision Transformer.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image-to-image tasks with overlapping patches.
    """
    def __init__(self, img_size=224, patch_size=16, stride=8, in_chans=3, out_chans=3, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, stride, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.out_chans = out_chans
        self.patch_size = patch_size

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_chans * patch_size ** 2)
        #self.out_conv = nn.ConvTranspose2d(out_chans, out_chans, kernel_size=patch_size, stride=patch_size)
        #self.out_conv = nn.ConvTranspose2d(81, out_chans, kernel_size=4, stride=4)
        self.out_conv = nn.ConvTranspose2d(self.patch_embed.n_patches, out_chans, kernel_size=patch_size,
                                           stride=2, padding=patch_size//2-1)


    def forward(self, x):
        #print(x.shape)
        B = x.shape[0]
        x = self.patch_embed(x)
        #print(x.shape)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 1:]  # Remove the class token
        x = self.proj(x)
        #x = x.reshape(B, self.out_chans, self.patch_embed.img_size, self.patch_embed.img_size)
        x = x.reshape(B, -1, self.patch_size, self.patch_size)
        x = self.out_conv(x)

        return x.permute(0, 2, 3, 1)


class CLIPVisionTransformer(nn.Module):
    """
    Vision Transformer model for image-to-image tasks with overlapping patches.
    """
    def __init__(self, img_size=224, patch_size=16, stride=8, in_chans=3, out_chans=3, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., llm=None):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, stride, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.out_chans = out_chans
        self.patch_size = patch_size
        self.llm = llm

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_chans * patch_size ** 2)

        # TODO: Figure out how to automate this (if possible, may need to be hard-coded?)
        self.out_conv = nn.ConvTranspose2d(self.patch_embed.n_patches+1, out_chans, kernel_size=patch_size,
                                           stride=2, padding=patch_size//2-1)

        # Hard-coded For CLIP
        #self.sentence_proj = nn.Linear(384, embed_dim//2)
        input_embed_dim = 768 if(self.llm == 'all-mpnet-base-v2') else 384
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(input_embed_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )
        #self.x_proj = nn.Linear(82*embed_dim, embed_dim//2)
        self.x_proj = nn.Sequential(
                                 nn.Linear((self.patch_embed.n_patches+1)*embed_dim, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )


    def forward(self, x, sentence_embeddings, clip=False, return_embedding=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        sentence_emb = self.sentence_proj(sentence_embeddings)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)

        x_emb = self.x_proj(x.flatten(1,2))
        x_emb = F.normalize(x_emb, p=2, dim=-1)

        if(return_embedding):
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        if(clip):
            cross_corr = x_emb @ sentence_emb.T
            return cross_corr
        else:
            embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1).detach()
            x = torch.cat((x, embedding), dim=1)

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 1:]  # Remove the class token
        x = self.proj(x)
        #x = x.reshape(B, self.out_chans, self.patch_embed.img_size, self.patch_embed.img_size)
        x = x.reshape(B, -1, self.patch_size, self.patch_size)
        x = self.out_conv(x)

        return x.permute(0, 2, 3, 1)

