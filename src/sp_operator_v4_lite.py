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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [B, H, W, D]
        """
        residual = x

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

        x = residual + self.drop_path(x)
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


class SphericalCrossOperator(nn.Module):
    """
    Query attends to operator tokens
    显存稳定版本（KV 只构图一次）
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        num_layers=2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": norm_layer(dim),
                "norm_kv": norm_layer(dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                "mlp": Mlp(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                )
            })
            for _ in range(num_layers)
        ])

        self.norm = norm_layer(dim)

    def forward(self, query, tokens, chunk_size=512):
        """
        query:  [B, N, D]
        tokens: [B, L+1, D]
        """
        B, N, D = query.shape
        outputs = []

        # ===============================
        # ✅ 核心修复：KV 只 norm 一次
        # ===============================
        kv_norm = self.layers[0]["norm_kv"](tokens)

        for i in range(0, N, chunk_size):
            q = query[:, i:i + chunk_size]  # [B, c, D]

            for layer in self.layers:
                q_norm = layer["norm_q"](q)

                attn_out, _ = layer["attn"](
                    q_norm,
                    kv_norm,
                    kv_norm,
                    need_weights=False,
                )

                q = q + attn_out
                q = q + layer["mlp"](q)

            outputs.append(q)

        out = torch.cat(outputs, dim=1)  # [B, N, D]
        return self.norm(out)

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
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.in_shape = in_shape
        self.out_shape = out_shape if out_shape is not None else in_shape
        self.num_layers = num_layers
        
        # self.conv = nn.Sequential(
        #     SphereConv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, kernel_size=1),
        # )

        # 构建多层球面注意力 + MLP 结构
        if self.in_shape is not None and self.out_shape is not None:
            # 使用球面注意力
            dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)] if num_layers > 1 else [drop_path]
            
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer = nn.ModuleDict({
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
                })
                self.layers.append(layer)
        else:
            # 如果没有形状信息，使用标准的多头注意力
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "norm1": norm_layer(dim),
                    "attn": nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=num_heads,
                        batch_first=True,
                    ),
                    "norm2": norm_layer(dim),
                    "mlp": Mlp(
                        in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        out_features=dim,
                    ),
                    "drop_path": DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                })
                for _ in range(num_layers)
            ])
        
        # 输出投影层
        if self.out_dim != dim:
            self.proj_out = nn.Linear(dim, self.out_dim)
            self.norm_out = norm_layer(self.out_dim)
        else:
            self.proj_out = nn.Identity()
            self.norm_out = nn.Identity()
        
        self.use_spherical_attn = (self.in_shape is not None and self.out_shape is not None)
    
    def forward(self, tokens):
        """
        tokens: [B, H, W, D] 或 [B, N, D]
        如果 use_spherical_attn=True，tokens 应该是 [B, H, W, D]
        否则 tokens 应该是 [B, N, D]
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
        
        # 应用多层处理
        if self.use_spherical_attn:
            # 球面注意力模式：需要转换为 [B, C, H, W] 格式
            for layer in self.layers:
                # 转换为 [B, C, H, W] 格式
                tokens_4d = tokens.permute(0, 3, 1, 2)  # [B, D, H, W]
                
                # 注意力层
                # LayerNorm 需要最后一个维度是 dim，所以先转换为 [B, H, W, D]
                tokens_4d = tokens_4d.permute(0, 2, 3, 1)  # [B, H, W, D]
                tokens_4d = layer["norm1"](tokens_4d)
                tokens_4d = tokens_4d.permute(0, 3, 1, 2)  # [B, D, H, W] 转回通道优先格式
                tokens_4d = tokens_4d + layer["attn"](tokens_4d)
                
                # MLP层（需要转换回 [B, H, W, D]）
                tokens_4d = tokens_4d.permute(0, 2, 3, 1)  # [B, H, W, D]

                if hasattr(layer, "ffn"):
                    tokens_4d = layer["ffn"](layer["norm2"](tokens_4d))
                
                tokens = tokens_4d  # [B, H, W, D]
        else:
            # 标准注意力模式：保持 [B, N, D] 格式
            for layer in self.layers:
                tokens = layer["norm1"](tokens)
                attn_out, _ = layer["attn"](tokens, tokens, tokens)
                tokens = tokens + layer["drop_path"](attn_out)
                tokens = layer["norm2"](tokens)
                tokens = tokens + layer["drop_path"](layer["mlp"](tokens))
        
        # 投影到输出维度
        if self.out_dim != self.dim:
            tokens = self.proj_out(tokens)
            tokens = self.norm_out(tokens)
        
        # 保持输出形状与输入形状一致
        if len(original_shape) == 3 and not self.use_spherical_attn:
            # 如果输入是 [B, N, D]，输出也应该是 [B, N, out_dim]
            pass  # tokens 已经是 [B, N, out_dim]
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
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
        use_residual=True,
    ):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, dim))

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

        q = self.query.expand(B, -1, -1)      # [B,1,D]
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
        x_proj = x @ self.kernel  # [N, 512]
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
        grid_size=(32, 64),high_gird=(64,128), pde_weight=0.0001, fourier_weight=1.0,latent_dim=1024, emb_dim=1024, dec_emb_dim=768, dec_num_heads=16, dec_depth=1, num_mlp_layers=1, out_dim=5, eps=1e5, layer_norm_eps=1e-5, embedding_type="latlon", chunk_size=512
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
        
        self.grid_size = grid_size
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
        )
        
        # Spherical set operator for query points and operator tokens interaction
        self.spherical_operator = SphericalCrossOperator(
            dim=self.dec_emb_dim,
            num_heads=self.dec_num_heads,
            mlp_ratio=self.mlp_ratio,
            num_layers=self.dec_depth,
        )
        
        self.spatial_norm = SpatialNorm(eps=self.layer_norm_eps)
        self.fusion_norm = nn.LayerNorm(self.dec_emb_dim, eps=self.layer_norm_eps)

        # 使用 SphericalSparseOperator 替代 head
        # 创建自定义的 norm_layer 函数
        def create_norm_layer(dim):
            return nn.LayerNorm(dim, eps=self.layer_norm_eps)
        
        self.head = SphericalSparseOperator(
            dim=self.dec_emb_dim,
            num_heads=self.dec_num_heads,
            out_dim=self.out_dim,
            in_shape=(self.high_grid[0].shape[0], self.high_grid[1].shape[0]),  # (H, W) - 确保是元组
            out_shape=(self.high_grid[0].shape[0], self.high_grid[1].shape[0]),  # (H, W) - 确保是元组
            mlp_ratio=self.mlp_ratio,
            num_layers=self.dec_depth,
            drop_path=drop_path,
            norm_layer=create_norm_layer,
        )

        # Create low grid and latents
        n_x, n_y = self.grid_size[0], self.grid_size[1]
        xx, yy = torch.meshgrid(n_x, n_y, indexing="ij")
        self.grid = torch.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
        self.latents = nn.Parameter(torch.randn(len(n_x) * len(n_y), self.latent_dim))

        #high
        n_x_high, n_y_high = high_gird[0], high_gird[1]
        xx_high, yy_high = torch.meshgrid(n_x_high,n_y_high, indexing="ij")
        self.grid_high = torch.hstack([xx_high.flatten()[:, None], yy_high.flatten()[:, None]])
    

        self.fourier_encoding = FourierEmbs(embed_scale=2 * np.pi, embed_dim=self.latent_dim)
        self.coord_fourier_grid_proj = MLP(self.latent_dim * 2, self.dec_emb_dim*2 ,self.dec_emb_dim)
       
        # --------------------------------------------------------------------------
    
    def coord_encoding_Fourier(self, b, coords):
        #coords.shape [H, W, 2] 
        
        H,W =  coords.shape[0], coords.shape[1]
        coords = coords.reshape(-1,2).to(self.latents.device).float()
        self.grid = self.grid.to(self.latents.device)
        fourier_embed = self.fourier_encoding(coords)
        d2 = ((coords[:, None, :] - self.grid[None, :, :]) ** 2).sum(dim=2).to(self.latents.device)
        # w = torch.exp(-self.eps * d2) / torch.exp(-self.eps * d2).sum(dim=1, keepdim=True).to(self.latents.device)
        w = torch.softmax(-self.eps * d2, dim=-1).to(self.latents.device)
        coords = (self.latents.T @ w.T).T
        
        coords = torch.cat([coords, fourier_embed], dim=-1)
        coords = self.coord_fourier_grid_proj(coords)
        coords = self.coord_norm(coords)

        coords = coords.unsqueeze(0).expand(b, -1, -1)  
        coords = coords.reshape(b, len(self.high_grid[0]), len(self.high_grid[1]), -1)
        
        return coords   #coords: [B, H, W, L]
        
    #---------------------------------------------------------------------------------
    # Operator: Spherical set operator for query points and operator tokens interaction
    def Operator(self, x, coords):
        # x: [B, L, D]
        # coords: [B, H, W, D]

        B, H, W, D = coords.shape
        N = H * W
        coords = coords.reshape(B, N, D)

        # prepend global token
        z_g = self.global_operator_token(x)     # [B,1,D]
        tokens = torch.cat([z_g, x], dim=1)     # [B,L+1,D]
        
        coords_refined = self.spherical_operator(
            coords,
            tokens,
            chunk_size=self.chunk_size
        )                                       # [B,N,D]

        coords_out = self.fusion_norm(coords_refined)
        # Reshape to [B, H, W, D] for SphericalSparseOperator
        coords_out = coords_out.reshape(B, H, W, self.dec_emb_dim)
        # Spatial normalization
        coords_out = self.spatial_norm(coords_out)
        # 使用 SphericalSparseOperator 进行球面稀疏注意力和维度变换
        coords_out = self.head(coords_out)  # [B, H, W, out_dim] 
        return coords_out

       
#---------------------------------------------------------------------------------


    def forward(self, embed_U, x, y, res, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        # x = x[:,0,:]

        embed_U = embed_U.squeeze(1)
        embed_U = self.embedmlp(embed_U)
        expectation_c = self.coord_encoding_Fourier(x.shape[0], self.grid_high)  #[B , H*W , L]

        preds = self.Operator(embed_U, expectation_c).permute(0,3,1,2) 
        preds = preds + res
        # + F.interpolate(x[:,torch.tensor([3, 4, 5, 8, 35, 40, 51]).type(torch.long).to(x.device),:], size=(64, 128), mode='bicubic', align_corners=False) #[B, H*W, Vo]
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, embed_U, x, y, res, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(embed_U, x, y, res, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
