import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Unfold, Fold

from .oformer import SpatialTemporalEncoder2D
from sentence_transformers import SentenceTransformer

# helpers
# Thank you Anthony

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
        
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
    
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
    
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., out_channels=1):
        super().__init__()
        self.channels = channels
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print("CHANNELS: {}".format(channels))
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.unpatchify = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.num_features = dim

        # Don't do pooling
        #self.pool = pool 

        # Project down to correct number of channels
        self.proj_down = nn.Sequential(
                             nn.Linear(4, 16),           # Not sure why this is hard-coded to 4.
                             nn.SiLU(),
                             nn.Linear(16, 16),
                             nn.SiLU(),
                             nn.Linear(16, out_channels)
        )
        self.to_latent = nn.Identity()
        self.channels = channels

    
    #def __repr__(self):
    #    return f'vit'
    
    def forward(self, img, features=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        # Remove class token
        x = x[:,:-1]
        x = self.unpatchify(x)
        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.proj_down(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        return x.unsqueeze(-1)


class OverlapViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., out_channels=1):
        super().__init__()
        self.channels = channels
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        print("CHANNELS: {}".format(channels))
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.rearranger = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.unfolder = Unfold(kernel_size=patch_height, dilation=1, padding=0, stride=patch_height//2)

        self.to_patch_embedding = nn.Sequential(
            #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            #Unfold(kernel_size=patch_height, dilation=1, padding=0, stride=patch_height//2),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, out_channels*dim),
            nn.LayerNorm(out_channels*dim),
        )

        # From unfold documentation
        num_patches = int(((image_size + 2*0 - (1 * (patch_height - 1)) - 1)/(patch_height//2))+1)**2
        print(num_patches)

        self.unpatchify = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)
        )
        self.folder = Fold(output_size=(image_size, image_size), kernel_size=patch_height, stride=patch_height//2)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, out_channels*dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels*dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(out_channels*dim, depth, heads, dim_head, mlp_dim, dropout)
        self.num_features = dim

        # Don't do pooling
        #self.pool = pool 

        # Project down to correct number of channels
        self.proj_down = nn.Sequential(
                             nn.Linear(4, 16),           # Not sure why this is hard-coded to 4.
                             nn.SiLU(),
                             nn.Linear(16, 16),
                             nn.SiLU(),
                             nn.Linear(16, out_channels)
        )
        self.to_latent = nn.Identity()
        self.channels = channels

    
    #def __repr__(self):
    #    return f'vit'
    
    def forward(self, img, features=False):

        #print("\n\nIMG SHAPE: {}".format(img.shape))

        rearange = self.rearranger(img)
        #print("REARRANGED: {}".format(rearange.shape))

        img = self.unfolder(img)#.permute(0,2,1)
        #print("FOLDED: {}".format(img.shape))
        #print("CHECK: {}\n\n".format((unfold == rearange).all()))
        #raise

        x = self.to_patch_embedding(img.permute(0,2,1))
        b, n, _ = x.shape
        
        #print("POST EMBEDDING: {}".format(x.shape))
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #print(x.shape, cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        # Remove class token
        x = x[:,:-1]
        #print()
        #print("POST TRANSFORMER: {}".format(x.shape))
        x = self.folder(x.permute(0,2,1))
        #print("POST UNFOLDING: {}".format(x.shape))
        #print()
        #return x.unsqueeze(-1)#.permute(0,2,3,1)
        #x = self.unpatchify(x)
        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.proj_down(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        return x.unsqueeze(-1)


class CLIPViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., llm=None, out_channels=1):
        super().__init__()
        self.channels = channels
        self.llm = llm
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        #patch_dim = channels * patch_height * patch_width
        #assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        #
        #self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #    nn.LayerNorm(patch_dim),
        #    nn.Linear(patch_dim, dim),
        #    nn.LayerNorm(dim),
        #)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.unpatchify = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.num_features = dim
        self.pool = pool 
        self.to_latent = nn.Identity()

        # Hard-coded For CLIP
        input_embed_dim = 768 if(self.llm == 'all-mpnet-base-v2') else 384
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(input_embed_dim, 2*dim),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, 2*dim),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, dim),
                                 nn.ReLU(),
                                 #nn.Linear(dim, dim//2)
                                 nn.Linear(dim, dim)
                                 #nn.Linear(2*dim, dim)
        )

        # Can also try OFormer here...
        # What is 64...? 
        #self.x_proj = nn.Sequential(
        #                         nn.Linear((64+1)*dim, 2*dim),
        #                         nn.SiLU(),
        #                         nn.Linear(2*dim, 2*dim),
        #                         nn.SiLU(),
        #                         nn.Linear(2*dim, dim),
        #                         nn.SiLU(),
        #                         nn.Linear(dim, dim//2)
        #                         #nn.Linear(dim, dim)
        #)

        # OFormer encoder
        self.x_proj = SpatialTemporalEncoder2D(input_channels=channels, heads=4, in_emb_dim=32,
                                          out_seq_emb_dim=dim//2, depth=4)
        self.proj_down = nn.Sequential(
                nn.Linear(image_size*image_size, 128),
                nn.SiLU(),
                nn.Linear(128,64),
                nn.SiLU(),
                nn.Linear(64,1)
        )

        self.final_proj_down = nn.Sequential(
                             nn.Linear(4, 16),           # Not sure why this is hard-coded to 4.
                             nn.SiLU(),
                             nn.Linear(16, 16),
                             nn.SiLU(),
                             nn.Linear(16, out_channels)
        )
        self.pretrained = False


    #def __repr__(self):
    #    return f'vit'

    
    def forward(self, img, sentence_embeddings, clip=False, return_embedding=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Add sentence embeddings here
        sentence_emb = self.sentence_proj(sentence_embeddings) # Lets also try with/without

        # Get data embedding
        #if(isinstance(self.x_proj, SpatialTemporalEncoder2D)):
        #    # Definitely not ideal... can rework if results are better
        #    coeffs = img[:,-5:]
        #    grid = img[:,-7:-5]
        #    data = img[:,:-7]
        #    data = torch.cat((data, coeffs), dim=1).flatten(2,3).permute(0,2,1)
        #    grid = grid.flatten(2,3).permute(0,2,1)

        #    x_emb = self.x_proj(data, input_pos=grid).permute(0,2,1)
        #    x_emb = self.proj_down(x_emb)[...,0]
        #else:
        #    x_emb = self.x_proj(x.flatten(1,2))

        ### Normalize embeddings
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
        #x_emb = F.normalize(x_emb, p=2, dim=-1)

        if(return_embedding):
            ##sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
            ##x_emb = F.normalize(x_emb, p=2, dim=-1)
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        if(clip):
            # Normalize embeddings
            ##sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
            ##x_emb = F.normalize(x_emb, p=2, dim=-1)
            cross_corr = x_emb @ sentence_emb.T
            return cross_corr
        else:

            #embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)
            embedding = sentence_emb.unsqueeze(1).clone()

            embedding = embedding.detach() if(self.pretrained) else embedding
            x = torch.cat((x, embedding), dim=1)
            #x = torch.cat((x, sentence_emb.unsqueeze(1)), dim=1)

        # Transformer forward
        x = self.transformer(x)
        
        # Remove class token
        x = x[:,:-2]
        x = self.unpatchify(x)

        # No pooling.
        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.final_proj_down(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        return x.unsqueeze(-1)


    def finished_pretraining(self):
        self.pretrained = True


class LLMCLIPViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64,
                 dropout = 0., emb_dropout = 0., llm=None, device='cuda'):
        super().__init__()
        self.llm = llm
        self.llm_model = SentenceTransformer(llm, device=device)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.unpatchify = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.num_features = dim
        self.pool = pool 
        self.to_latent = nn.Identity()

        # Hard-coded For CLIP
        input_embed_dim = 768 if(self.llm == 'all-mpnet-base-v2') else 384
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(input_embed_dim, 2*dim),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, 2*dim),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, dim),
                                 nn.ReLU(),
                                 nn.Linear(dim, dim//2)
                                 #nn.Linear(2*dim, dim)
        )

        # Can also try OFormer here...
        # What is 64...? 
        #self.x_proj = nn.Sequential(
        #                         nn.Linear((64+1)*dim, 2*dim),
        #                         nn.SiLU(),
        #                         nn.Linear(2*dim, 2*dim),
        #                         nn.SiLU(),
        #                         nn.Linear(2*dim, dim),
        #                         nn.SiLU(),
        #                         nn.Linear(dim, dim//2)
        #                         #nn.Linear(dim, dim)
        #)

        # OFormer encoder
        self.x_proj = SpatialTemporalEncoder2D(input_channels=channels, heads=4, in_emb_dim=32,
                                          out_seq_emb_dim=dim//2, depth=4)
        self.proj_down = nn.Sequential(
                nn.Linear(image_size*image_size, 128),
                nn.SiLU(),
                nn.Linear(128,64),
                nn.SiLU(),
                nn.Linear(64,1)
        )
        self.pretrained = False

    def __repr__(self):
        return f'vit'


    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm_model.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm_model.device)
        output = self.llm_model(tokenized_sentences)['sentence_embedding']
        return output

    
    def forward(self, img, sentence_embeddings, clip=False, return_embedding=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Add sentence embeddings here
        if(not(self.pretrained)):
            sentence_emb = self._llm_forward(sentence_embeddings)
        else:
            sentence_emb = sentence_embeddings.clone()
        sentence_emb = self.sentence_proj(sentence_emb) # Lets also try with/without
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)

        if(isinstance(self.x_proj, SpatialTemporalEncoder2D)):
            # Definitely not ideal... can rework if results are better
            coeffs = img[:,-5:]
            grid = img[:,-7:-5]
            data = img[:,:-7]
            data = torch.cat((data, coeffs), dim=1).flatten(2,3).permute(0,2,1)
            grid = grid.flatten(2,3).permute(0,2,1)

            x_emb = self.x_proj(data, input_pos=grid).permute(0,2,1)
            x_emb = self.proj_down(x_emb)[...,0]
        else:
            x_emb = self.x_proj(x.flatten(1,2))
        x_emb = F.normalize(x_emb, p=2, dim=-1)

        if(return_embedding):
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        if(clip):
            cross_corr = x_emb @ sentence_emb.T
            return cross_corr
        else:
            embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach() -> only detach if we've done pretraining
            embedding = embedding.detach() if(self.pretrained) else embedding
            x = torch.cat((x, embedding), dim=1)

        # Transformer forward
        x = self.transformer(x)
        
        # Remove class token
        x = x[:,:-2]
        x = self.unpatchify(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        return x.unsqueeze(-1)

    def finished_pretraining(self):
        self.pretrained = True

