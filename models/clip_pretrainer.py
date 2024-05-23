import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from .oformer import SpatialTemporalEncoder2D

class LLMPretraining(nn.Module):
    def __init__(self, llm, embed_dim=32, im_size=32, initial_step=10, device='cuda'):
        super().__init__()
        self.llm = SentenceTransformer(llm, device=device)
        self.llm_dim = 768 if(llm == 'all-mpnet-base-v2') else 384
        self.embed_dim = embed_dim
        self.im_size = im_size
        self.initial_step = initial_step
        self.num_channels = 1 # Maybe generalize this in the future

        # Sentence projector
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(self.llm_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Data projector -> Could try OFormer encoder here
        self.x_proj = nn.Sequential(
                                 nn.Linear((self.initial_step*self.num_channels+2)*self.im_size*self.im_size, self.embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Upsampling -> Could try OFormer decoder here
        self.proj_up = nn.Sequential(
                                 nn.Linear(32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, self.im_size*self.im_size)
        ).to(device)

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm.device)
        output = self.llm(tokenized_sentences)['sentence_embedding']
        return output

    def forward(self, x, grid, sentence, clip=False, return_embedding=False):
        # Get data embedding
        x = torch.cat((x, grid), dim=-1)
        x_emb = self.x_proj(x.flatten(1,3))

        # Embed and project sentences
        sentence_emb = self._llm_forward(sentence)
        sentence_emb = self.sentence_proj(sentence_emb)
        if(return_embedding):
            cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        elif(clip):
            cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            return cross_corr
        else:
            stacked_emb = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)
            return self.proj_up(stacked_emb).reshape((x.shape[0], 1, self.im_size, self.im_size))


class OFormerLLMPretraining(nn.Module):
    def __init__(self, llm, embed_dim=32, im_size=32, initial_step=10, device='cuda', detach=False):
        super().__init__()
        self.llm = SentenceTransformer(llm, device=device)
        self.llm_dim = 768 if(llm == 'all-mpnet-base-v2') else 384
        self.embed_dim = embed_dim
        self.im_size = im_size
        self.initial_step = initial_step
        self.num_channels = 1 # Maybe generalize this in the future
        self.detach = detach

        # Sentence projector
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(self.llm_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Data projector -> Could try OFormer encoder here
        #self.x_proj = nn.Sequential(
        #                         nn.Linear((self.initial_step*self.num_channels+2)*self.im_size*self.im_size, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim//2)
        #).to(device)

        self.x_proj = SpatialTemporalEncoder2D(input_channels=self.num_channels*self.initial_step+2, heads=4, in_emb_dim=128,
                                               out_seq_emb_dim=self.embed_dim//2, depth=4)
        self.x_proj_down = nn.Sequential(
                                 nn.Linear(32*32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 64),
                                 nn.SiLU(),
                                 nn.Linear(64,1)
        ).to(device)

        # Upsampling -> Could try OFormer decoder here
        self.proj_up = nn.Sequential(
                                 nn.Linear(self.embed_dim, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, self.im_size*self.im_size)
        ).to(device)

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm.device)
        output = self.llm(tokenized_sentences)['sentence_embedding']
        return output

    def forward(self, x, grid, sentence, clip=False, return_embedding=False):
        # Get data embedding
        #x = torch.cat((x, grid), dim=-1)
        if(len(x.shape) == 4):
            x = x.flatten(1,2)
        if(len(grid.shape) == 4):
            grid = grid.flatten(1,2)
        x_emb = self.x_proj(x, grid).transpose(1,2)
        x_emb = self.x_proj_down(x_emb)[...,0]
        x_emb = F.normalize(x_emb, p=2, dim=-1)

        # Embed and project sentences
        sentence_emb = self._llm_forward(sentence)
        sentence_emb = self.sentence_proj(sentence_emb)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
        if(return_embedding):
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        elif(clip):
            #print(x_emb.shape, sentence_emb.shape)
            cross_corr = x_emb @ sentence_emb.T
            #cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))#.softmax(dim=1)
            #print(cross_corr.shape)
            return cross_corr
        else:
            stacked_emb = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach()
            if(self.detach): # May help reduce overfitting
                stacked_emb = stacked_emb.detach()
            return self.proj_up(stacked_emb).reshape((x.shape[0], 1, self.im_size, self.im_size))


class OFormerInputLLMPretraining(nn.Module):
    def __init__(self, llm, embed_dim=32, im_size=32, initial_step=10, device='cuda', detach=False):
        super().__init__()
        self.llm = SentenceTransformer(llm, device=device)
        self.llm_dim = 768 if(llm == 'all-mpnet-base-v2') else 384
        self.embed_dim = embed_dim
        self.im_size = im_size
        self.initial_step = initial_step
        self.num_channels = 1 # Maybe generalize this in the future
        self.detach = detach

        # Sentence projector
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(self.llm_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Data projector -> Could try OFormer encoder here
        #self.x_proj = nn.Sequential(
        #                         nn.Linear((self.initial_step*self.num_channels+2)*self.im_size*self.im_size, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim//2)
        #).to(device)

        self.x_proj = SpatialTemporalEncoder2D(input_channels=self.num_channels*self.initial_step+2, heads=4, in_emb_dim=128,
                                               out_seq_emb_dim=self.embed_dim//2, depth=4)
        self.x_proj_down = nn.Sequential(
                                 nn.Linear(32*32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 64),
                                 nn.SiLU(),
                                 nn.Linear(64,1)
        ).to(device)

        # Upsampling -> Could try OFormer decoder here
        self.proj_up = nn.Sequential(
                                 nn.Linear(self.embed_dim, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, self.im_size*self.im_size)
        ).to(device)

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        print(sentence[0])
        tokenized_sentences = self.llm.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm.device)
        print(tokenized_sentences['input_ids'][0])
        print(tokenized_sentences['token_type_ids'][0])
        print(tokenized_sentences['attention_mask'][0])
        output = self.llm(tokenized_sentences)['sentence_embedding']
        print(output.shape)
        raise
        return output

    def forward(self, x, grid, sentence, clip=False, return_embedding=False):
        # Get data embedding
        #x = torch.cat((x, grid), dim=-1)
        if(len(x.shape) == 4):
            x = x.flatten(1,2)
        if(len(grid.shape) == 4):
            grid = grid.flatten(1,2)
        #print(self.x_proj)
        #print(x.shape, grid.shape)
        x_emb = self.x_proj(x, grid).transpose(1,2)
        x_emb = self.x_proj_down(x_emb)[...,0]
        x_emb = F.normalize(x_emb, p=2, dim=-1)

        # Embed and project sentences
        sentence_emb = self._llm_forward(sentence)
        sentence_emb = self.sentence_proj(sentence_emb)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
        if(return_embedding):
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        elif(clip):
            #print(x_emb.shape, sentence_emb.shape)
            cross_corr = x_emb @ sentence_emb.T
            #cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))#.softmax(dim=1)
            #print(cross_corr.shape)
            return cross_corr
        else:
            stacked_emb = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach()
            if(self.detach): # May help reduce overfitting
                stacked_emb = stacked_emb.detach()
            return self.proj_up(stacked_emb).reshape((x.shape[0], 1, self.im_size, self.im_size))


class CLIPPretraining(nn.Module):
    def __init__(self, llm, embed_dim=32, im_size=32, initial_step=10, device='cuda'):
        super().__init__()
        self.llm = SentenceTransformer(llm, device=device)
        self.llm_dim = 768 if(llm == 'all-mpnet-base-v2') else 384
        self.embed_dim = embed_dim
        self.im_size = im_size
        self.initial_step = initial_step
        self.num_channels = 1 # Maybe generalize this in the future

        # Sentence projector
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(self.llm_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Data projector -> Could try OFormer encoder here
        self.x_proj = nn.Sequential(
                                 nn.Linear((self.initial_step*self.num_channels+2)*self.im_size*self.im_size, self.embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Upsampling -> Could try OFormer decoder here
        self.proj_up = nn.Sequential(
                                 nn.Linear(32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, self.im_size*self.im_size)
        ).to(device)

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm.device)
        output = self.llm(tokenized_sentences)['sentence_embedding']
        return output

    def forward(self, x, grid, sentence, clip=False, return_embedding=False):
        # Get data embedding
        x = torch.cat((x, grid), dim=-1)
        x_emb = self.x_proj(x.flatten(1,3))

        # Embed and project sentences
        sentence_emb = self.sentence_proj(sentence.cuda())
        if(return_embedding):
            cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        elif(clip):
            cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            return cross_corr
        else:
            stacked_emb = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1).detach()
            return self.proj_up(stacked_emb).reshape((x.shape[0], 1, self.im_size, self.im_size))


class OFormerCLIPPretraining(nn.Module):
    def __init__(self, llm, embed_dim=32, im_size=32, initial_step=10, device='cuda'):
        super().__init__()
        self.llm = SentenceTransformer(llm, device=device)
        self.llm_dim = 768 if(llm == 'all-mpnet-base-v2') else 384
        self.embed_dim = embed_dim
        self.im_size = im_size
        self.initial_step = initial_step
        self.num_channels = 1 # Maybe generalize this in the future

        # Sentence projector
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(self.llm_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim//2)
        ).to(device)

        # Data projector -> Could try OFormer encoder here
        #self.x_proj = nn.Sequential(
        #                         nn.Linear((self.initial_step*self.num_channels+2)*self.im_size*self.im_size, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim),
        #                         nn.SiLU(),
        #                         nn.Linear(self.embed_dim, self.embed_dim//2)
        #).to(device)
        self.x_proj = SpatialTemporalEncoder2D(input_channels=self.num_channels*self.initial_step+2, heads=4, in_emb_dim=128,
                                               out_seq_emb_dim=self.embed_dim//2, depth=4)
        self.x_proj_down = nn.Sequential(
                                 nn.Linear(32*32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 64),
                                 nn.SiLU(),
                                 nn.Linear(64,1)
        ).to(device)

        # Upsampling -> Could try OFormer decoder here
        self.proj_up = nn.Sequential(
                                 nn.Linear(self.embed_dim, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, self.im_size*self.im_size)
        ).to(device)

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm.tokenize(sentence)
        for key, val in tokenized_sentences.items():
            tokenized_sentences[key] = val.to(self.llm.device)
        output = self.llm(tokenized_sentences)['sentence_embedding']
        return output

    def forward(self, x, grid, sentence, clip=False, return_embedding=False):
        # Get data embedding
        #x = torch.cat((x, grid), dim=-1)
        #x_emb = self.x_proj(x.flatten(1,3))
        if(len(x.shape) == 4):
            x = x.flatten(1,2)
        if(len(grid.shape) == 4):
            grid = grid.flatten(1,2)
        x_emb = self.x_proj(x, grid).transpose(1,2)
        x_emb = self.x_proj_down(x_emb)[...,0]
        x_emb = F.normalize(x_emb, p=2, dim=-1)

        # Embed and project sentences
        #sentence_emb = self.sentence_proj(sentence.cuda())
        sentence_emb = self._llm_forward(sentence)
        sentence_emb = self.sentence_proj(sentence_emb)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
        if(return_embedding):
            #cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        elif(clip):
            #cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            cross_corr = x_emb @ sentence_emb.T
            return cross_corr
        else:
            stacked_emb = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)
            return self.proj_up(stacked_emb).reshape((x.shape[0], 1, self.im_size, self.im_size))
