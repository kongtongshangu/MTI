# -*- coding: utf-8 -*-
import torchvision
import torch.nn.functional as F

from copy import deepcopy
from operator import mul
from functools import reduce
import math
import numpy as np

import os

os.environ['TORCH_HOME'] = './pretrained_model'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    logits: shape [batch_size, num_categories]
    temperature: controls the sharpness of the distribution
    hard: if True, return a one-hot vector, else return a soft probability distribution
    """
    gumbel_noise = sample_gumbel(logits.size())
    y = logits + gumbel_noise
    y = F.softmax(y / temperature, dim=-1)
    if hard:
        _, max_index = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(dim=-1, index=max_index, value=1.0)
        return y_hard - y.detach() + y
    # return torch.rand(logits.size()[0], 2).cuda()
    return y


def LinearWeightedMovingAverage(L):
    l = len(L)
    if l == 0:
        return torch.zeros(768, 768)
    weights = [i for i in range(1, l + 1)]
    return sum([idx * weight for idx, weight in zip(L, weights)]) / sum(weights)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., type='nofuse'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if type == 'fuse':
            self.q, self.k, self.v = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(
                dim, dim, bias=qkv_bias)
        self.type = type
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.first_batch = True

    def forward(self, x, ft=False):
        if ft == True:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn

        B, N, C = x.shape

        if self.type == 'fuse':
            if self.first_batch:
                self.q.weight.data, self.q.bias.data = self.qkv.weight.data[:self.dim, :], self.qkv.bias.data[:self.dim]
                self.k.weight.data, self.k.bias.data = self.qkv.weight.data[self.dim:self.dim * 2,
                                                       :], self.qkv.bias.data[self.dim:self.dim * 2]
                self.v.weight.data, self.v.bias.data = self.qkv.weight.data[self.dim * 2:, :], self.qkv.bias.data[
                                                                                               self.dim * 2:]
                self.first_batch = False
            qkv = torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1).reshape(B, N, 3, self.num_heads,
                                                                               C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        sattn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.type == 'fuse':
            vis_attn = sattn.mean(1).detach()
        else:
            vis_attn = attn.mean(1).mean(1).detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, vis_attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, type='nofuse'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, type=type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None, attn_scale=None, use_tta=None, ft=False):
        if modality == None:
            output, attn = self.attn(self.norm1(x), ft=ft)
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':

            # output, attn = self.attn(self.norm1_a(x), attn_scale=attn_scale, use_tta=use_tta, ft=ft)
            output, attn = self.attn(self.norm1_a(x))
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            # output, attn = self.attn(self.norm1_v(x), attn_scale=attn_scale, use_tta=use_tta, ft=ft)
            output, attn = self.attn(self.norm1_v(x))
            x = x + self.drop_path(output)
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x, attn


# the finetuned CAV-MAE model
class CAVMAEFT_OURS(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, tr_pos=True, args=None):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        # print('Use norm_pix_loss: ', norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches,
                                                                           self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim),
                                        requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim),
                                        requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in
             range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in
             range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, type='fuse')
             for i in range(12 - modality_specific_depth)])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.args = args
        self.initialize_weights()

        self.num_experts = 5
        self.a_adaptor = nn.Parameter(torch.zeros(self.num_experts, embed_dim, embed_dim), requires_grad=True)
        self.v_adaptor = nn.Parameter(torch.zeros(self.num_experts, embed_dim, embed_dim), requires_grad=True)

        self.v_router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_experts)
        )
        self.a_router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_experts)
        )
        # self.weight_adaptor = nn.Parameter(torch.Tensor([0, 0]))
        self.gate_a_adaptor = nn.Parameter(torch.tensor([0.]))
        self.gate_v_adaptor = nn.Parameter(torch.tensor([0.]))

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches / 8),
                                              cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5),
                                              int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a, _ = blk(a, ft=True)

            for blk in self.blocks_v:
                v, _ = blk(v, ft=True)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x, _ = blk(x, ft=True)
            x = self.norm(x)

            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u)  # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a')  # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u)  # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v')  # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    def forward_adapt(self, a, v, mode, TSA=False):
        bs = a.shape[0]
        NB = 1 * bs
        # multi-modal TTA
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for idx, blk in enumerate(self.blocks_a):
                a, a_attn = blk(a)

            for idx, blk in enumerate(self.blocks_v):
                v, v_attn = blk(v)


            tau = 1.0
            score_a = torch.sigmoid(self.gate_a_adaptor / tau).to(a.device)
            score_v = torch.sigmoid(self.gate_v_adaptor / tau).to(v.device)

            a_ctx = a.mean(dim=1)
            v_ctx = v.mean(dim=1)

            a_weights = F.softmax(self.a_router(a_ctx), dim=-1)
            v_weights = F.softmax(self.v_router(v_ctx), dim=-1)

            a_dynamic_mat = torch.einsum('bk,kmn->bmn', a_weights, self.a_adaptor)
            v_dynamic_mat = torch.einsum('bk,kmn->bmn', v_weights, self.v_adaptor)

            eye_mat = torch.eye(768).to(a.device).unsqueeze(0)
            a_instance_adaptor = a_dynamic_mat + eye_mat
            v_instance_adaptor = v_dynamic_mat + eye_mat

            a_adapted = torch.matmul(a, a_instance_adaptor)
            v_adapted = torch.matmul(v, v_instance_adaptor)

            a = score_a * a_adapted + (1 - score_a) * a
            v = score_v * v_adapted + (1 - score_v) * v

            x = torch.cat((v, a), dim=1)
            for blk in self.blocks_u:
                x, attn = blk(x, ft=False)

            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)

            if TSA:
                # return x, attn, choice_probs
                return x, attn, (a_weights, v_weights), a, v
            else:
                return x, attn

    def forward_eval(self, a, v, mode, TSA=False, Traverse=False):
        bs = a.shape[0]
        # multi-modal TTA
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for idx, blk in enumerate(self.blocks_a):
                a, a_attn = blk(a)

            for idx, blk in enumerate(self.blocks_v):
                v, v_attn = blk(v)

            if TSA:
                score_a = torch.sigmoid(self.gate_a_adaptor).item()
                score_v = torch.sigmoid(self.gate_v_adaptor).item()

                a_ctx = a.mean(dim=1)
                a_weights = F.softmax(self.a_router(a_ctx), dim=-1)
                a_dynamic_mat = torch.einsum('bk,kmn->bmn', a_weights, self.a_adaptor)
                a_instance_adaptor = a_dynamic_mat + torch.eye(768).to(a.device).unsqueeze(0)
                a_adapted = torch.matmul(a, a_instance_adaptor)

                v_ctx = v.mean(dim=1)
                v_weights = F.softmax(self.v_router(v_ctx), dim=-1)
                v_dynamic_mat = torch.einsum('bk,kmn->bmn', v_weights, self.v_adaptor)
                v_instance_adaptor = v_dynamic_mat + torch.eye(768).to(v.device).unsqueeze(0)
                v_adapted = torch.matmul(v, v_instance_adaptor)

                if Traverse:
                    a = torch.cat([a, a_adapted], dim=0)
                    v = torch.cat([v_adapted, v], dim=0)
                else:
                    choice_probs = torch.softmax(self.weight_adaptor.expand(bs, -1) / self.args.gumbel_softmax_tau, dim=1)
                    selection = choice_probs[:, 0].view(-1, 1, 1)
                    a = selection * a + (1 - selection) * a_adapted
                    v = (1 - selection) * v + selection * v_adapted

            x = torch.cat((v, a), dim=1)
            for blk in self.blocks_u:
                x, attn = blk(x, ft=False)
            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)

            if TSA:
                return x, attn, None
            else:
                return x, attn

        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a, _ = blk(a)

            v = torch.zeros((a.shape[0], 1, a.shape[2])).to(a.device)
            x = torch.cat((a, v), dim=1)
            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                x, _ = blk(x, 'a')

            x = self.norm_a(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x, a

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v, _ = blk(v)

            a = torch.zeros((v.shape[0], 317, v.shape[2])).to(v.device)
            x = torch.cat((a, v), dim=1)
            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                x, _ = blk(x, 'v')
            x = self.norm_v(x)
            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x, v