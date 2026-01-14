# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache
from re import X

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_,DropPath,Mlp, Attention
import sys
# from climax.parallelpatchembed import ParallelVarPatchEmbed
import torch.nn.functional as F
from torch_harmonics import NeighborhoodAttentionS2
sys.path.append("/home/hunter/workspace/climate/climate_predict/")
from sphere_conv import SphereConv2d
from torch_harmonics.examples.models._layers import MLP, DropPath, LayerNorm, SequencePositionEmbedding, SpectralPositionEmbedding, LearnablePositionEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union


def latlon_to_xyz(coords, radius=1.0):
    """
    将经纬度转换为 3D 笛卡尔坐标（参考 sphere_posEmb.py）
    这样可以避免在经纬度空间中直接计算距离的问题（极点奇异性、经度周期性等）
    
    Args:
        coords: [..., 2] -> (lat, lon) 弧度
        radius: 球面半径，默认为 1.0
    
    Returns:
        xyz: [..., 3] -> (x, y, z) 3D 笛卡尔坐标
    """
    lat = coords[..., 0]
    lon = coords[..., 1]
    # 气象习惯：lat 从 pi/2 到 -pi/2
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def compute_geodesic_distance(coords1, coords2, radius=1.0, coords_in_degrees=False):
    """
    计算球面上两点之间的测度距离（geodesic distance）
    参考 sphere_posEmb.py 的方法：先将经纬度转换为3D笛卡尔坐标，然后计算3D空间中的距离
    
    Args:
        coords1: [N, 2] 第一组坐标 (lat, lon)
        coords2: [M, 2] 第二组坐标 (lat, lon)
        radius: 球面半径，默认为 1.0（单位球面）
        coords_in_degrees: bool, 如果为 True，坐标单位为度数，需要转换为弧度
    
    Returns:
        distances: [N, M] 距离矩阵，单位与 radius 相同
    """
    # 如果需要，将度数转换为弧度
    if coords_in_degrees:
        coords1 = torch.deg2rad(coords1)
        coords2 = torch.deg2rad(coords2)
    
    # 转换为3D笛卡尔坐标
    coords1_3d = latlon_to_xyz(coords1, radius=radius)  # [N, 3]
    coords2_3d = latlon_to_xyz(coords2, radius=radius)  # [M, 3]
    
    # 在3D空间中计算欧式距离（这就是球面上的测度距离）
    # (N, 1, 3) 与 (1, M, 3) 广播 -> (N, M, 3) -> (N, M)
    d2 = ((coords1_3d[:, None, :] - coords2_3d[None, :, :]) ** 2).sum(dim=2)
    distances = torch.sqrt(d2)
    
    return distances


def compute_geodesic_distance_squared(coords1, coords2, radius=1.0, coords_in_degrees=False):
    """
    计算球面上两点之间的测度距离的平方
    参考 sphere_posEmb.py 的方法：使用3D笛卡尔坐标计算
    
    Args:
        coords1: [N, 2] 第一组坐标 (lat, lon)
        coords2: [M, 2] 第二组坐标 (lat, lon)
        radius: 球面半径，默认为 1.0
        coords_in_degrees: bool, 如果为 True，坐标单位为度数
    
    Returns:
        distances_squared: [N, M] 距离平方矩阵
    """
    # 如果需要，将度数转换为弧度
    if coords_in_degrees:
        coords1 = torch.deg2rad(coords1)
        coords2 = torch.deg2rad(coords2)
    
    # 转换为3D笛卡尔坐标
    coords1_3d = latlon_to_xyz(coords1, radius=radius)  # [N, 3]
    coords2_3d = latlon_to_xyz(coords2, radius=radius)  # [M, 3]
    
    # 在3D空间中计算欧式距离的平方（这就是球面上测度距离的平方）
    # (N, 1, 3) 与 (1, M, 3) 广播 -> (N, M, 3) -> (N, M)
    d2 = ((coords1_3d[:, None, :] - coords2_3d[None, :, :]) ** 2).sum(dim=2)
    
    return d2


class SphericalCrossOperatorS2(nn.Module):
    """
    Cross Attention Operator using Spherical Neighborhood Attention and Spectral PE.
    Input/Output interface remains [B, N, D] for compatibility.
    """
    def __init__(
        self,
        dim,
        num_heads,
        in_shape=(32, 64),   # KV shape (Low Res)
        out_shape=(64, 128), # Q shape (High Res)
        mlp_ratio=4.0,
        num_layers=1,
        norm_layer=nn.LayerNorm,
        theta_cutoff=None,    # Optional: override neighborhood radius
        use_spherical_attn=True  # True: NeighborhoodAttentionS2, False: MultiheadAttention
    ):
        super().__init__()
        
        self.dim = dim
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.num_heads = num_heads
        self.use_spherical_attn = use_spherical_attn

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if use_spherical_attn:
                # S2 Neighborhood Attention
                attn_module = NeighborhoodAttentionS2(
                    in_channels=dim,
                    in_shape=in_shape,    # Low Res Grid
                    out_shape=out_shape,  # High Res Grid
                    num_heads=num_heads,
                    grid_in="equiangular",
                    grid_out="equiangular",
                    theta_cutoff=theta_cutoff,
                    bias=True,
                    # k_channels/out_channels 默认等于 in_channels
                )
            else:
                # Standard Multihead Attention
                attn_module = nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    bias=True,
                    batch_first=False,  # 使用 (seq_len, batch, embed_dim) 格式
                )
            
            layer = nn.ModuleDict({
                "norm_q": norm_layer(dim),
                "norm_kv": norm_layer(dim),
                "attn": attn_module,
                "norm_mlp": norm_layer(dim),
                "mlp": Mlp(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                )
            })
            self.layers.append(layer)
      
    def forward(self, query, tokens):
        """
        query:  [B, H*W, D] (High Res Target)
        tokens: [B, h*w, D] (Low Res Source)
        """
        B, N_q, D = query.shape
        B, N_kv, D = tokens.shape
        
        # 校验形状
        H_q, W_q = self.out_shape
        h_kv, w_kv = self.in_shape
        if self.use_spherical_attn:
            assert N_q == H_q * W_q, f"Query len {N_q} mismatch with shape {self.out_shape}"
        # assert N_kv == h_kv * w_kv , f"Token len {N_kv} mismatch with shape {self.in_shape}"

        # # -------------------------------------------------------
        # # 1. 注入 Spectral Position Embedding
        # # -------------------------------------------------------
        # # PE 模块已经处理了 [B, N, D] 的输入情况
        # query = self.pos_embed_q(query)   # [B, H*W, D]
        # tokens = self.pos_embed_kv(tokens) # [B, h*w, D]

        for layer in self.layers:
            # ---------------------------------------------------
            # 2. Attention Block (Norm -> Reshape -> Attn -> Add)
            # ---------------------------------------------------
            
            # Pre-Norm
            q_norm = layer["norm_q"](query)    # [B, Nq, D]
            kv_norm = layer["norm_kv"](tokens) # [B, Nkv, D]
            
            if self.use_spherical_attn:
                # Reshape to Spatial [B, D, H, W] for NeighborhoodAttentionS2
                # transpose(1, 2) -> [B, D, N] -> view -> [B, D, H, W]
                q_spatial = q_norm.transpose(1, 2).view(B, D, H_q, W_q)
                k_spatial = kv_norm.transpose(1, 2).view(B, D, h_kv, w_kv)
                v_spatial = k_spatial # Use same features for Key and Value
                
                # Neighborhood Cross Attention
                # Output will be [B, D, H_q, W_q]
                attn_out_spatial = layer["attn"](
                    query=q_spatial,
                    key=k_spatial,
                    value=v_spatial
                )
                
                # Reshape back to Sequence [B, Nq, D]
                attn_out = attn_out_spatial.view(B, D, N_q).transpose(1, 2)
            else:
                # Standard Multihead Attention
                # MultiheadAttention expects (seq_len, batch, embed_dim) format
                # q_norm: [B, Nq, D] -> [Nq, B, D]
                # kv_norm: [B, Nkv, D] -> [Nkv, B, D]
                q_seq = q_norm.transpose(0, 1)  # [Nq, B, D]
                kv_seq = kv_norm.transpose(0, 1)  # [Nkv, B, D]
                
                # Multihead Attention
                attn_out_seq, _ = layer["attn"](
                    query=q_seq,
                    key=kv_seq,
                    value=kv_seq,
                    need_weights=False
                )
                
                # Reshape back to [B, Nq, D]
                attn_out = attn_out_seq.transpose(0, 1)  # [B, Nq, D]
            
            # Residual Connection
            query = query + attn_out
            
            # ---------------------------------------------------
            # 3. MLP Block (Norm -> MLP -> Add)
            # ---------------------------------------------------
            # MLP usually works on [B, N, D] naturally
            q_norm_mlp = layer["norm_mlp"](query)
            query = query + layer["mlp"](q_norm_mlp)

        return query

         
class SphericalFFN(nn.Module):
    """
    Spherical Geometry-aware Feed-Forward Network (SG-FFN)

    Acts as a geometry-consistent replacement for the token-wise MLP
    in Transformer blocks operating on spherical grids.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        kernel_size=3,
        drop_path=0.0,
        act_layer=nn.GELU,
        use_layerscale=True,
        layerscale_init=1e-6,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # 1) Depthwise spherical convolution (spatial mixing)
        self.dwconv = SphereConv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        # 2) Channel expansion + projection (channel mixing)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # 3) LayerScale (optional but recommended)
        if use_layerscale:
            self.gamma = nn.Parameter(layerscale_init * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x):
        """
        x: [B, H, W, D]
        """
        # [B, H, W, D] -> [B, D, H, W]
        x = x.permute(0, 3, 1, 2)

        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # [B, D, H, W] -> [B, H, W, D]
        x = x.permute(0, 2, 3, 1)

        if self.gamma is not None:
            x = self.gamma * x

        return x


class SpatialNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: [B, H, W, D]
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + self.eps)

class SphericalSparseOperator(nn.Module):
    """
    球面稀疏操作符 - 优化版本，支持多层处理、维度变换和球面注意力
    """
    def __init__(
        self, 
        dim, 
        num_heads, 
        out_dim=None,
        in_shape=None,
        out_shape=None,
        mlp_ratio=4.0, 
        num_layers=2, 
        drop_path=0.1, 
        norm_layer=nn.LayerNorm,
        grid_in="equiangular",
        grid_out="equiangular",
        chunk_size=10,
        psi_col_idx=None,
        psi_roff_idx=None,
        attn_scale_init=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.in_shape = in_shape
        self.out_shape = out_shape if out_shape is not None else in_shape
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.psi_col_idx = psi_col_idx
        self.psi_roff_idx = psi_roff_idx
        self.attn_scale_init = attn_scale_init

        # 构建多层球面注意力 + MLP 结构
        if self.in_shape is not None and self.out_shape is not None:
            # 使用球面注意力
            dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)] if num_layers > 1 else [drop_path]
            
            self.sparse_attns = nn.ModuleList()
            # self.gate_attn = nn.ParameterList()
            # self.gate_ffn = nn.ParameterList()
            for i in range(num_layers):
                self.sparse_attns.append(nn.ModuleDict({
                    "norm1": norm_layer(dim),
                    "attn": NeighborhoodAttentionS2(
                        in_channels=dim,
                        num_heads=num_heads,
                        in_shape=self.in_shape,
                        out_shape=self.out_shape,
                        grid_in=grid_in,
                        grid_out=grid_out,
                    ),
                    "norm2": norm_layer(dim),
                    "ffn": SphericalFFN(
                        dim,
                        mlp_ratio=mlp_ratio,
                        kernel_size=3,
                        drop_path=dpr[i],
                    ),
                }))
                # self.gate_attn.append(
                #     nn.Parameter(torch.logit(torch.tensor(0.5)))
                # )
                # self.gate_ffn.append(
                #     nn.Parameter(torch.logit(torch.tensor(0.5)))
                # )
       
        self.ffn = myMlp(num_layers=num_layers, hidden_dim=dim, out_dim=dim)
        
        # 融合前的标准化层（可选，用于稳定训练）
        self.fusion_norm_mlp = norm_layer(dim)
        self.fusion_norm_attn = norm_layer(dim)

        # 🔑 关键：attention 是"慢变量"
        self.register_buffer(
            "attn_scale",
            torch.tensor(self.attn_scale_init),
            persistent=False
        )

        # 输出投影层
        if self.out_dim != dim:
            self.proj_out = myMlp(num_layers=num_layers, hidden_dim=dim, out_dim=self.out_dim)
        else:
            self.proj_out = nn.Identity()
        
        self.use_spherical_attn = (self.in_shape is not None and self.out_shape is not None)
    
    def forward(self, tokens, attn_scale=None):
        """
        tokens: [B, H, W, D] 或 [B, N, D]
        如果 use_spherical_attn=True，tokens 应该是 [B, H, W, D]
        否则 tokens 应该是 [B, N, D]
        attn_scale: Attention scale factor for dynamic scaling.
        """
        original_shape = tokens.shape
        B = tokens.shape[0]
        
        # 处理输入形状
        if len(original_shape) == 3:
            # [B, N, D] -> [B, H, W, D] 或保持 [B, N, D]
            B, N, D = tokens.shape
            if self.use_spherical_attn:
                H, W = self.in_shape
                if N != H * W:
                    raise ValueError(f"Token数量 {N} 与形状 {self.in_shape} 不匹配 (需要 {H*W})")
                tokens = tokens.reshape(B, H, W, D)
        elif len(original_shape) == 4:
            # [B, H, W, D] - 已经是正确格式
            B, H, W, D = tokens.shape
        else:
            raise ValueError(f"不支持的输入形状: {original_shape}")
        
        # 在形状处理完成后克隆，确保 x 与 tokens 形状一致
        x = tokens.clone()
        
        # 使用传入的 attn_scale，如果没有则使用默认值
        if attn_scale is None:
            attn_scale = self.attn_scale.item() if hasattr(self, 'attn_scale') else 0.1
        else:
            attn_scale = float(attn_scale)
        
        # 应用多层处理
        if self.use_spherical_attn:
            # 球面注意力模式：需要转换为 [B, C, H, W] 格式
            attn_out = None  # 初始化 attn_out
            for i, layer in enumerate(self.sparse_attns):
                # 转换为 [B, C, H, W] 格式
                tokens_4d = tokens.permute(0, 3, 1, 2)  # [B, D, H, W]
                
                # 注意力层
                # LayerNorm 需要最后一个维度是 dim，所以先转换为 [B, H, W, D]
                tokens_4d = tokens_4d.permute(0, 2, 3, 1)  # [B, H, W, D]
                tokens_4d = layer["norm1"](tokens_4d)
                tokens_4d = tokens_4d.permute(0, 3, 1, 2)  # [B, D, H, W] 转回通道优先格式
                # gate_attn = torch.sigmoid(self.gate_attn[i])
                # breakpoint()
                # 应用 attn_scale 缩放 attention 输出
                attn_out = layer["attn"](tokens_4d)  # [B, D, H, W]
                tokens_4d = tokens_4d + attn_scale * attn_out

                # MLP层（需要转换回 [B, H, W, D]）
                tokens_4d = tokens_4d.permute(0, 2, 3, 1)  # [B, H, W, D]

                if hasattr(layer, "ffn"):
                    # gate_ffn = torch.sigmoid(self.gate_ffn[i])
                    # tokens_4d = tokens_4d + gate_ffn * layer["ffn"](layer["norm2"](tokens_4d))
                    tokens_4d = tokens_4d + layer["ffn"](layer["norm2"](tokens_4d))
                
                tokens = tokens_4d  # [B, H, W, D]
            
            # 在球面注意力模式下，使用 ffn 和 attn_out 进行融合
            out_mlp = self.ffn(x)  # x 是 [B, H, W, D]，out_mlp 也是 [B, H, W, D]
            # attn_out 是 [B, D, H, W]，需要转换为 [B, H, W, D]
            attn_out_4d = attn_out.permute(0, 2, 3, 1)  # [B, H, W, D]
            
            # 标准化后再融合（有助于稳定训练和平衡两个分支的贡献）
            out_mlp_norm = self.fusion_norm_mlp(out_mlp)
            attn_out_norm = self.fusion_norm_attn(attn_out_4d)
            tokens = out_mlp_norm + attn_scale * attn_out_norm
        else:
            # 标准注意力模式：需要 [B, N, D] 格式
            # 如果tokens是4维的，先reshape成3维
            if len(tokens.shape) == 4:
                B, H, W, D = tokens.shape
                tokens = tokens.reshape(B, H * W, D)  # [B, H*W, D]
                # 同时更新 x 的形状
                x = x.reshape(B, H * W, D)
            
            B, N, D = tokens.shape
            
            attn_out = None  # 初始化 attn_out
            for layer in self.sparse_attns:
                tokens = layer["norm1"](tokens)
                
                # 使用chunk处理注意力计算以节省内存
                attn_outputs = []
                # for i in range(0, N, self.chunk_size):
                # chunk_tokens = tokens[:, i:i + self.chunk_size]  # [B, chunk_size, D]
                attn_out, _ = layer["attn"](tokens, tokens, tokens)  # [B, N, D]
                # attn_outputs.append(attn_out)
                # attn_out = torch.cat(attn_outputs, dim=1)  # [B, N, D]
                
                tokens = tokens + attn_out
                tokens = layer["norm2"](tokens)
                tokens = tokens + layer["mlp"](tokens)
            
            # 在标准注意力模式下，使用 ffn 和 attn_out 进行融合
            out_mlp = self.ffn(x)  # x 是 [B, N, D]，out_mlp 也是 [B, N, D]
            # attn_out 已经是 [B, N, D]，不需要 permute
            
            # 标准化后再融合（有助于稳定训练和平衡两个分支的贡献）
            out_mlp_norm = self.fusion_norm_mlp(out_mlp)
            attn_out_norm = self.fusion_norm_attn(attn_out)
            tokens = out_mlp_norm + attn_scale * attn_out_norm
            
        # 投影到输出维度
        if self.out_dim != self.dim:
            tokens = self.proj_out(tokens) # mlp
        
        # 保持输出形状与输入形状一致
        if not self.use_spherical_attn:
            # 标准注意力模式
            if len(original_shape) == 4:
                # 如果原始输入是4维的，需要reshape回4维
                B, H, W, _ = original_shape
                tokens = tokens.reshape(B, H, W, self.out_dim)  # [B, H, W, out_dim]
            # 如果原始输入是3维的，tokens已经是 [B, N, out_dim]，保持不变
        elif len(original_shape) == 3 and self.use_spherical_attn:
            # 如果输入是 [B, N, D] 但使用了球面注意力，需要转换回 [B, N, out_dim]
            B, H, W, D = tokens.shape
            tokens = tokens.reshape(B, H * W, D)
        # 如果输入是 [B, H, W, D]，输出也保持 [B, H, W, out_dim]
        
        return tokens

class GlobalOperatorToken(nn.Module):
    """
    从球面 operator latent 中抽取全局条件 token
    U: [B, L, D] -> z_g: [B, 1, D]
    使用稀疏注意力减少计算量
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
        use_residual=True,
        num_query=1,
    ):
        super().__init__()

        self.num_query = num_query
        self.query = nn.Parameter(torch.randn(1, num_query, dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.use_residual = use_residual

    def forward(self, U):
        """
        U: [B, L, D]
        return:
            z_g: [B, 1, D]
        """
        B, L, D = U.shape

        q = self.query.expand(B, self.num_query, -1)      # [B,num_query,D]
        kv = self.norm_kv(U)

        z_g, _ = self.attn(q, kv, kv)         # [B,1,D]

        if self.use_residual:
            z_g = z_g + q

        return self.norm_out(z_g)

class GlobalOperatorReadout(nn.Module):
    """
    输出:
      - global token
      - augmented operator tokens
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.readout = GlobalOperatorToken(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, U, return_augmented=True):
        """
        U: [B, L, D]
        """
        z_g = self.readout(U)  # [B,1,D]

        if return_augmented:
            U_aug = torch.cat([z_g, U], dim=1)  # [B, L+1, D]
            return z_g, U_aug

        return z_g

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x
    
class FourierEmbs(nn.Module):
    def __init__(self, embed_scale, embed_dim):
        super(FourierEmbs, self).__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.kernel = nn.Parameter(torch.randn(2, self.embed_dim // 2))

    def forward(self, x):
        
        # 应用傅里叶变换
        x_proj = x @ self.kernel  # [N, 2] @ [2, 512] -> [N, 512]
        y = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # [N, 1024]
        return y

class myMlp(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim, layer_norm_eps=1e-5):
        super(myMlp, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_norm_eps = layer_norm_eps
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim, eps=self.layer_norm_eps)
            ) for _ in range(self.num_layers)
        ])
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        x = self.output_layer(x)
        return x

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self, 
        default_vars,
        # encoder,
        time_range=4,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        encoder_depth=8,
        fuse_decoder_depth=2,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        low_gird=(32, 64),high_gird=(64,128), pde_weight=0.0001, fourier_weight=1.0,latent_dim=1024, emb_dim=1024, dec_emb_dim=768, dec_num_heads=16, dec_depth=1, num_mlp_layers=1, out_dim=5, eps=1e5, layer_norm_eps=1e-5, embedding_type="latlon", chunk_size=10, num_global_operator_token=16, theta_cutoff=None, use_spherical_attn=True
    ):
        super().__init__()

    # Feature Extracer
    # --------------------------------------------------------------------------
       # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        self.time_range = time_range
        
        self.embedmlp = MLP(in_features=emb_dim, out_features=dec_emb_dim)
       
        self.high_lamda = 0.0001
        
        self.low_gird = low_gird
        self.high_grid = high_gird
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dec_emb_dim = dec_emb_dim
        self.dec_num_heads = dec_num_heads
        self.dec_depth = dec_depth
        self.num_mlp_layers = num_mlp_layers
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.eps = eps
        self.layer_norm_eps = layer_norm_eps
        self.embedding_type = embedding_type
        self.chunk_size = chunk_size  # Chunk size for processing query points to reduce memory
        
        self.coord_norm = nn.LayerNorm(self.dec_emb_dim, eps=self.layer_norm_eps)
       
        self.global_operator_token = GlobalOperatorToken(
            dim=self.dec_emb_dim,
            num_heads=self.dec_num_heads,
            dropout=drop_rate,
            use_residual=True,
            num_query=num_global_operator_token,
        )
        
        # Spherical set operator for query points and operator tokens interaction
        self.spherical_operator = SphericalCrossOperatorS2(
            dim=self.dec_emb_dim,
            num_heads=self.dec_num_heads,
            mlp_ratio=self.mlp_ratio,
            in_shape=(self.low_gird[0].shape[0], self.low_gird[1].shape[0]),  # (H, W) - 确保是元组
            out_shape=(self.high_grid[0].shape[0], self.high_grid[1].shape[0]),  # (H, W) - 确保是元组
            theta_cutoff=theta_cutoff, 
            use_spherical_attn=use_spherical_attn,
        )
        
        self.spatial_norm = SpatialNorm(eps=self.layer_norm_eps)
        self.fusion_norm = nn.LayerNorm(self.dec_emb_dim, eps=self.layer_norm_eps)

        # 使用 SphericalSparseOperator 替代 head
        # 创建自定义的 norm_layer 函数
        def create_norm_layer(dim):
            return nn.LayerNorm(dim, eps=self.layer_norm_eps)
        
        # self.head = SphericalSparseOperator(
        #     dim=self.dec_emb_dim,
        #     num_heads=self.dec_num_heads,
        #     out_dim=self.out_dim,
        #     in_shape=(self.high_grid[0].shape[0], self.high_grid[1].shape[0]),  # (H, W) - 确保是元组
        #     out_shape=(self.high_grid[0].shape[0], self.high_grid[1].shape[0]),  # (H, W) - 确保是元组
        #     mlp_ratio=self.mlp_ratio,
        #     num_layers=self.dec_depth,
        #     drop_path=drop_path,
        #     norm_layer=create_norm_layer,
        #     chunk_size=self.chunk_size,
        # )

        self.head = myMlp(num_layers=self.num_mlp_layers, hidden_dim=self.dec_emb_dim, out_dim=self.out_dim, layer_norm_eps=self.layer_norm_eps)

        # Create low grid and latents
        n_x, n_y = low_gird[0], low_gird[1]
        xx, yy = torch.meshgrid(n_x, n_y, indexing="ij")
        self.grid = torch.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
        self.latents = nn.Parameter(torch.randn(len(n_x) * len(n_y), self.latent_dim))

        #high
        n_x_high, n_y_high = high_gird[0], high_gird[1]
        xx_high, yy_high = torch.meshgrid(n_x_high,n_y_high, indexing="ij")
        self.grid_high = torch.hstack([xx_high.flatten()[:, None], yy_high.flatten()[:, None]])
    
        self.fourier_encoding = FourierEmbs(embed_scale=2 * np.pi, embed_dim=self.latent_dim)
        self.coord_fourier_grid_proj = MLP(self.latent_dim * 2, self.dec_emb_dim*2 ,self.dec_emb_dim)
        
        # 测度距离参数
        self.use_geodesic_distance = True  # 是否使用测度距离
        self.sphere_radius = 1.0  # 球面半径（单位球面）
        self.coords_in_degrees = True  # 坐标单位是否为度数（False 表示弧度）
       
        # --------------------------------------------------------------------------
    
    def _compute_position_encoding(self, query_coords, reference_grid=None, use_geodesic=True, sphere_radius=1.0, coords_in_degrees=True):
        """
        通用的位置编码计算函数
        
        Args:
            query_coords: [N, 2] 查询坐标（需要编码的坐标点），单位：弧度 (lat, lon)
            reference_grid: [M, 2] 参考网格（用于计算距离和加权），单位：弧度 (lat, lon)
                           如果为 None 则使用 query_coords 自身
            use_geodesic: bool, 是否使用测度距离（geodesic distance），默认 True
            sphere_radius: float, 球面半径，用于测度距离计算，默认 1.0
            coords_in_degrees: bool, 坐标单位是否为度数（False 表示弧度）
        Returns:
            encoded: [N, dec_emb_dim] 编码后的位置特征
        """
        device = query_coords.device if hasattr(query_coords, 'device') else self.latents.device
        query_coords = query_coords.float()  # [N, 2]
        
        # 如果没有指定参考网格，使用查询坐标自身（用于低分辨率自编码）
        if reference_grid is None:
            reference_grid = query_coords
        
        reference_grid = reference_grid.to(device).float()  # [M, 2]
        latents = self.latents.to(device)  # [M, latent_dim]
        
        # 1. Fourier 编码
        fourier_embed = self.fourier_encoding(query_coords)  # [N, latent_dim]
        
        # 2. 计算距离：query_coords 到 reference_grid 的距离
        if use_geodesic:
            # 使用测度距离（geodesic distance）
            d2 = compute_geodesic_distance_squared(
                query_coords, 
                reference_grid, 
                radius=sphere_radius,
                coords_in_degrees=self.coords_in_degrees
            )  # [N, M]
        else:
            # 使用欧式距离（原始方法）
            d2 = ((query_coords[:, None, :] - reference_grid[None, :, :]) ** 2).sum(dim=2)  # [N, M]
        
        # 3. 使用 softmax 权重（距离越近权重越大）
        w = torch.softmax(-self.eps * d2, dim=-1)  # [N, M]
        
        # 4. 使用权重加权 latents
        weighted_latents = (latents.T @ w.T).T  # [latent_dim, M] @ [M, N] -> [N, latent_dim]
        
        # 5. 拼接加权后的 latents 和 Fourier 编码
        encoded_input = torch.cat([weighted_latents, fourier_embed], dim=-1)  # [N, latent_dim * 2]
        
        # 6. 映射到解码器维度
        encoded = self.coord_fourier_grid_proj(encoded_input)  # [N, dec_emb_dim]
        encoded = self.coord_norm(encoded)
        
        return encoded
    
    def get_low_res_pe(self, b, device):
        """
        低分辨率位置编码，使用与高分辨率相同的编码方式
        计算低分辨率网格点之间的距离，并使用 softmax 权重加权 latents
        """
        # self.grid 是初始化时保存的 [low_h * low_w, 2]
        low_grid = self.grid.to(device).float()  # [L, 2]
        
        # 使用通用编码函数，reference_grid=None 表示使用自身作为参考（自编码）
        low_pe = self._compute_position_encoding(
            low_grid, 
            reference_grid=None,
            use_geodesic=self.use_geodesic_distance,
            sphere_radius=self.sphere_radius
        )  # [L, dec_emb_dim]
        
        return low_pe.unsqueeze(0).expand(b, -1, -1)  # [B, L, dec_emb_dim]

    def coord_encoding_Fourier(self, b, coords):
        """
        高分辨率位置编码
        计算高分辨率坐标到低分辨率网格的距离，并使用 softmax 权重加权 latents
        """
        # coords.shape [H*W, 2] 或 [H, W, 2]
        coords = coords.reshape(-1, 2).to(self.latents.device).float()  # [H*W, 2]
        self.grid = self.grid.to(self.latents.device)
        
        # 使用通用编码函数，reference_grid=self.grid 表示使用低分辨率网格作为参考
        coords_encoded = self._compute_position_encoding(
            coords, 
            reference_grid=self.grid,
            use_geodesic=self.use_geodesic_distance,
            sphere_radius=self.sphere_radius,
            coords_in_degrees=self.coords_in_degrees,
        )  # [H*W, dec_emb_dim]
        
        # Reshape 回空间维度
        coords_encoded = coords_encoded.unsqueeze(0).expand(b, -1, -1)  # [B, H*W, dec_emb_dim]
        coords_encoded = coords_encoded.reshape(b, len(self.high_grid[0]), len(self.high_grid[1]), -1)  # [B, H, W, dec_emb_dim]
        
        return coords_encoded
        
    #---------------------------------------------------------------------------------
    # Operator: Spherical set operator for query points and operator tokens interaction
    def Operator(self, x, coords):
        # x: [B, L, D]
        # coords: [B, H, W, D]

        B, H, W, D = coords.shape
        N = H * W
        coords = coords.reshape(B, N, D)

        # prepend global token
        # z_g = self.global_operator_token(x)     # [B,1,D]
        # tokens = torch.cat([z_g, x], dim=1)     # [B,num_global_operator_token + L,D]
        # coords = coords + z_g.unsqueeze(1)
        coords_spatial = self.spherical_operator(coords, x)    # [B,N,D]

        coords_normalized = self.fusion_norm(coords_spatial)
        # Reshape to [B, H, W, D] for SphericalSparseOperator
        coords_spatial = coords_normalized.reshape(B, H, W, self.dec_emb_dim)
        # Spatial normalization
        coords_spatial = self.spatial_norm(coords_spatial)
        # 使用 SphericalSparseOperator 进行球面稀疏注意力和维度变换
        coords_out = self.head(coords_spatial)  # [B, H, W, out_dim] 
        return coords_out

       
#---------------------------------------------------------------------------------


    def forward(self, embed_U, x, y, res, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            attn_scale: Attention scale factor for dynamic scaling.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        # x = x[:,0,:]

        embed_U = embed_U.squeeze(1)
        embed_U = self.embedmlp(embed_U)

        coords_high = self.coord_encoding_Fourier(x.shape[0], self.grid_high)  #[B , H*W , L]
        low_pe = self.get_low_res_pe(x.shape[0], x.device)

        token_low = embed_U + low_pe 

        preds = self.Operator(token_low, coords_high).permute(0,3,1,2)
        preds = preds + res
        
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, embed_U, x, y, res, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(embed_U, x, y, res, out_variables, metric=None, lat=lat)
        num_vars = len(out_variables)
        return [m(preds[:, :num_vars], y[:, :num_vars], transform, out_variables, lat, clim, log_postfix) for m in metrics]
