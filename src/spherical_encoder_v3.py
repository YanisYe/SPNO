import torch
import torch.nn as nn
import torch.nn.functional as F

from sphere_conv import SphereConv2d

class SphericalAvgPool(nn.Module):
    """
    纬度加权的球面平均池化（低通）
    """
    def __init__(self, lat_weights):
        super().__init__()
        # lat_weights: [H]，通常是 cos(lat)
        self.register_buffer("lat_weights", lat_weights.view(1, 1, -1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.lat_weights
        x_weighted = x * w
        return x_weighted.sum(dim=(-2, -1), keepdim=True) / w.sum()

class SphericalLaplacianPyramid(nn.Module):
    """
    真正的球面 Laplacian Pyramid
    
    - 每一层都是球面低通
    
    - 尺度由 kernel size / blur strength 控制
    
    - 所有尺度保持 [B, C, H, W]
    """
    def __init__(self, num_scales, in_channels):
        super().__init__()
        self.num_scales = num_scales
        self.lowpass_filters = nn.ModuleList()
        
        for s in range(num_scales):
            k = 3 + 2 * s   # 3,5,7,9,...
            self.lowpass_filters.append(
                SphereConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=in_channels,   # depthwise
                    bias=False
                )
            )

    def forward(self, x):
        """
        return:
        dict[s] = Laplacian band at scale s
        all in [B, C, H, W]
        """
        pyramid = {}
        low_prev = x
        
        for s in range(self.num_scales - 1):
            low = self.lowpass_filters[s](low_prev)
            pyramid[s] = low_prev - low
            low_prev = low
        
        pyramid[self.num_scales - 1] = low_prev
        return pyramid


# -----------------------------
# A.  球面卷积块（带残差连接，参考 ResBlock）
# -----------------------------
class SphereConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.nonlinearity = nn.GELU()
        
        # 第一个卷积：保持通道数不变
        self.conv1 = SphereConv2d(in_c, in_c, kernel_size=3, padding=1)
        # 对于1x1特征图，使用普通卷积
        self.conv1_1x1 = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)
        
        # 第二个卷积：改变通道数，bias=False（参考 ResBlock）
        self.conv2 = SphereConv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        # 对于1x1特征图，使用普通卷积
        self.conv2_1x1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        
        # 归一化层（在第二个卷积之后）- 使用 GroupNorm
        self.norm = nn.GroupNorm(num_groups=16, num_channels=out_c)
        
        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # 如果输入输出通道数不同，需要投影残差连接
        self.use_residual = (in_c == out_c)
        if not self.use_residual:
            # shortcut 使用普通 Conv2d，因为 1x1 卷积不需要球面卷积的特殊处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
                # shortcut 也使用 GroupNorm
                nn.GroupNorm(num_groups=16, num_channels=out_c)
            )

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        
        # 判断是否为1x1特征图或宽度为奇数
        use_1x1_conv = (H == 1 and W == 1) or (W % 2 != 0)
        
        if use_1x1_conv:
            # 对于1x1特征图，使用普通卷积
            x = self.conv1_1x1(x)
            x = self.nonlinearity(x)
            x = self.conv2_1x1(x)
        else:
            # 第一个卷积 + 激活
            x = self.conv1(x)
            x = self.nonlinearity(x)
            
            # 第二个卷积
            x = self.conv2(x)
        
        # 归一化（在第二个卷积之后，参考 ResBlock）
        x = self.norm(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 残差连接
        if self.use_residual:
            x = x + residual
        else:
            x = x + self.shortcut(residual)
        
        return x

class SphericalLaplacianConv(nn.Module):
    """
    一级结构：尺度 = first-class citizen
    """
    def __init__(self, in_c, out_c, num_scales, lat_weights, dropout=0.1, share_weights=False):
        super().__init__()
        self.pyramid = SphericalLaplacianPyramid(num_scales, in_c)

        if share_weights:
            block = SphereConvBlock(in_c, out_c, dropout)
            self.scale_blocks = nn.ModuleList([block for _ in range(num_scales)])
        else:
            self.scale_blocks = nn.ModuleList([
                SphereConvBlock(in_c, out_c, dropout)
                for _ in range(num_scales)
            ])

    def forward(self, x):
        # 1. 尺度分解
        pyramid = self.pyramid(x)
        B, C_orig, H_orig, W_orig = x.shape

        # 2. 每尺度独立建模
        outputs = []
        for s, block in enumerate(self.scale_blocks):
            z_s = block(pyramid[s])  # [B, C_out, H_s, W_s]
            # 确保所有尺度输出都有相同的空间维度
            if z_s.shape[2:] != (H_orig, W_orig):
                # 使用expand_as扩展到原始尺寸
                z_s = z_s.expand(B, z_s.shape[1], H_orig, W_orig)
            outputs.append(z_s)

        # 3. 输出为尺度堆叠（或 dict）
        # [B, S, C, H, W]
        return torch.stack(outputs, dim=1)

class ScaleTokenExtractor(nn.Module):
    def __init__(self, lat_weights):
        super().__init__()
        self.register_buffer("lat_weights", lat_weights.view(1, 1, -1, 1))

    def forward(self, z):
        # z: [B, S, C, H, W]
        w = self.lat_weights
        z_weighted = z * w.unsqueeze(1)   # broadcast 到 S
        token = z_weighted.sum(dim=(-2, -1)) / w.sum()
        # token: [B, S, C]
        return token

class ScaleRelativeBias(nn.Module):
    def __init__(self, num_scales, num_heads):
        super().__init__()
        self.bias = nn.Parameter(
            torch.zeros(num_heads, num_scales, num_scales)
        )

    def forward(self):
        return self.bias

class ScaleAwareAttention(nn.Module):
    def __init__(self, dim, num_scales, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.scale_bias = ScaleRelativeBias(num_scales, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, scale_tokens):
        """
        scale_tokens: [B, S, C]
        """
        x = self.norm(scale_tokens)

        attn_out, _ = self.scale_attn(
            x, x, x,
            attn_mask=None,
            need_weights=False
        )

        return scale_tokens + attn_out

class ScaleFiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_gamma = nn.Linear(dim, dim)
        self.to_beta = nn.Linear(dim, dim)

    def forward(self, z, scale_tokens):
        # z: [B, S, C, H, W]
        gamma = self.to_gamma(scale_tokens).unsqueeze(-1).unsqueeze(-1)
        beta  = self.to_beta(scale_tokens).unsqueeze(-1).unsqueeze(-1)
        return gamma * z + beta

# -----------------------------
# F. Full Encoder
# -----------------------------
class WeatherSphericalEncoder(nn.Module):
    def __init__(
        self,
        in_channels=54,
        lat_weights=None,
        num_scales=4,
        conv_dim=256,
        embed_dim=1024,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # 1. 球面 Laplacian Conv（一级尺度结构）
        self.spherical_conv = SphericalLaplacianConv(
            in_c=in_channels,
            out_c=conv_dim,
            num_scales=num_scales,
            lat_weights=lat_weights,
            dropout=dropout,
        )

        # 2. 尺度 token（用于跨尺度 attention）
        self.scale_token = ScaleTokenExtractor(lat_weights)
        self.scale_attn = ScaleAwareAttention(
            dim=conv_dim,
            num_scales=num_scales,
            num_heads=4,
        )
        self.scale_film = ScaleFiLM(conv_dim)

        # 3. 尺度融合（沿尺度维度做可学习的能量融合）
        # 每个像素位置，沿尺度维度做一次可学习的融合
        self.scale_fuse = nn.Conv3d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=(num_scales, 1, 1),
            bias=False
        )

        # 4. 像素级投影（每个像素映射到embedding维度）
        self.pixel_proj = nn.Conv2d(
            conv_dim,
            embed_dim,
            kernel_size=1
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, L, D] 其中 L = H × W (per-location latent embedding)
        """

        # ---- 1. 多尺度球面特征 ----
        z = self.spherical_conv(x)      # [B, S, C, H, W]

        # ---- 2. 跨尺度调制 ----
        scale_tokens = self.scale_token(z)          # [B, S, C]
        scale_tokens = self.scale_attn(scale_tokens)
        z = self.scale_film(z, scale_tokens)        # [B, S, C, H, W]

        # ---- 3. 尺度融合（保持空间维度）----
        # z: [B, S, C, H, W] -> [B, C, S, H, W]
        z = z.permute(0, 2, 1, 3, 4)
        # 沿尺度维度做可学习的融合
        z = self.scale_fuse(z).squeeze(2)  # [B, C, H, W]

        # ---- 4. 像素级投影到embedding维度 ----
        z = self.pixel_proj(z)  # [B, D, H, W]

        # ---- 5. Token化（只在最后reshape）----
        B, D, H, W = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, L, D], L = H × W

        return self.norm(z)

