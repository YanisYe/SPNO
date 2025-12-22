"""
球面物理约束损失函数
基于物理规律（散度、涡度、水汽守恒等）对模型预测进行约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
import os
import torch_harmonics as th

from torch_harmonics.quadrature import _precompute_latitudes

class PhysicsLossS2(nn.Module):
    """
    球面物理约束损失函数
    
    包含以下物理约束：
    1. 散度约束：风场的散度应该满足连续性方程（接近0）
    2. 涡度约束：涡度的物理一致性
    3. 水汽守恒：水汽的物理约束
    
    Parameters
    ----------
    nlat : int
        纬度方向的分辨率
    nlon : int
        经度方向的分辨率
    grid : str, optional
        网格类型，默认为 "equiangular"
    divergence_weight : float, optional
        散度损失的权重，默认为 1.0
    vorticity_weight : float, optional
        涡度损失的权重，默认为 0.5
    vapor_weight : float, optional
        水汽损失的权重，默认为 0.5
    use_divergence : bool, optional
        是否使用散度约束，默认为 True
    use_vorticity : bool, optional
        是否使用涡度约束，默认为 True
    use_vapor : bool, optional
        是否使用水汽约束，默认为 True
    """
    
    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str = "equiangular",
        divergence_weight: float = 1.0,
        vorticity_weight: float = 0.5,
        vapor_weight: float = 0.5,
        use_divergence: bool = True,
        use_vorticity: bool = True,
        use_vapor: bool = True,
    ):
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.divergence_weight = divergence_weight
        self.vorticity_weight = vorticity_weight
        self.vapor_weight = vapor_weight
        self.use_divergence = use_divergence
        self.use_vorticity = use_vorticity
        self.use_vapor = use_vapor
        
        # 获取球面积分权重
        _, q = _precompute_latitudes(nlat=nlat, grid=grid)
        q = q.reshape(-1, 1) * 2 * torch.pi / nlon
        q = q / torch.sum(q) / float(nlon)
        q = torch.tile(q, (1, nlon)).contiguous()
        q = q.to(torch.float32)
        
        self.register_buffer("quad_weights", q)
        
        # 设置FFT频率网格用于计算梯度
        l_phi = 2 * torch.pi  # 经度方向
        l_theta = torch.pi    # 纬度方向
        
        k_phi = torch.fft.fftfreq(nlon, d=l_phi / (2 * torch.pi * nlon))
        k_theta = torch.fft.fftfreq(nlat, d=l_theta / (2 * torch.pi * nlat))
        k_theta_mesh, k_phi_mesh = torch.meshgrid(k_theta, k_phi, indexing="ij")
        self.register_buffer("k_phi_mesh", k_phi_mesh)
        self.register_buffer("k_theta_mesh", k_theta_mesh)
    
    def compute_gradients(self, x: torch.Tensor) -> tuple:
        """
        使用FFT计算球面上的梯度
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch, nlat, nlon) 或 (batch, channels, nlat, nlon)
        
        Returns
        -------
        grad_lat : torch.Tensor
            纬度方向的梯度
        grad_lon : torch.Tensor
            经度方向的梯度
        """
        # 确保有批次维度
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # 如果是4D张量，对每个通道分别计算
        if x.dim() == 4:
            batch, channels, nlat, nlon = x.shape
            grad_lat_list = []
            grad_lon_list = []
            for c in range(channels):
                x_c = x[:, c, :, :]
                grad_lat_c = torch.fft.ifft2(1j * self.k_theta_mesh * torch.fft.fft2(x_c)).real
                grad_lon_c = torch.fft.ifft2(1j * self.k_phi_mesh * torch.fft.fft2(x_c)).real
                grad_lat_list.append(grad_lat_c)
                grad_lon_list.append(grad_lon_c)
            grad_lat = torch.stack(grad_lat_list, dim=1)
            grad_lon = torch.stack(grad_lon_list, dim=1)
        else:
            # 3D张量 (batch, nlat, nlon)
            grad_lat = torch.fft.ifft2(1j * self.k_theta_mesh * torch.fft.fft2(x)).real
            grad_lon = torch.fft.ifft2(1j * self.k_phi_mesh * torch.fft.fft2(x)).real
        
        return grad_lat, grad_lon
    
    def compute_divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        计算风场的散度
        
        散度 = ∂u/∂x + ∂v/∂y
        
        对于球面坐标：
        - x方向对应经度方向（lon）
        - y方向对应纬度方向（lat）
        
        Parameters
        ----------
        u : torch.Tensor
            u风分量，形状为 (batch, nlat, nlon)
        v : torch.Tensor
            v风分量，形状为 (batch, nlat, nlon)
        
        Returns
        -------
        divergence : torch.Tensor
            散度，形状为 (batch, nlat, nlon)
        """
        # 计算梯度
        dudx, _ = self.compute_gradients(u)  # ∂u/∂x (经度方向)
        _, dvdy = self.compute_gradients(v)  # ∂v/∂y (纬度方向)
        
        # 散度 = ∂u/∂x + ∂v/∂y
        divergence = dudx + dvdy
        return divergence
    
    def compute_vorticity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        计算风场的涡度
        
        涡度 = ∂v/∂x - ∂u/∂y
        
        Parameters
        ----------
        u : torch.Tensor
            u风分量，形状为 (batch, nlat, nlon)
        v : torch.Tensor
            v风分量，形状为 (batch, nlat, nlon)
        
        Returns
        -------
        vorticity : torch.Tensor
            涡度，形状为 (batch, nlat, nlon)
        """
        # 计算梯度
        dudy, _ = self.compute_gradients(u)  # ∂u/∂y (纬度方向)
        _, dvdx = self.compute_gradients(v)  # ∂v/∂x (经度方向)
        
        # 涡度 = ∂v/∂x - ∂u/∂y
        vorticity = dvdx - dudy
        return vorticity
    
    def compute_divergence_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        计算散度约束损失
        
        理想情况下，风场的散度应该接近0（连续性方程）
        损失 = |散度|^2 的球面积分
        
        Parameters
        ----------
        u : torch.Tensor
            u风分量，形状为 (batch, nlat, nlon)
        v : torch.Tensor
            v风分量，形状为 (batch, nlat, nlon)
        
        Returns
        -------
        loss : torch.Tensor
            散度损失（标量）
        """
        divergence = self.compute_divergence(u, v)
        # 计算散度的平方
        divergence_sq = torch.square(divergence)
        # 球面积分
        loss = torch.sum(divergence_sq * self.quad_weights, dim=(-2, -1))
        return torch.mean(loss)
    
    def compute_vorticity_consistency_loss(self, u_pred: torch.Tensor, v_pred: torch.Tensor,
                                          u_target: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """
        计算涡度一致性损失
        
        预测和目标的涡度应该一致
        
        Parameters
        ----------
        u_pred : torch.Tensor
            预测的u风分量
        v_pred : torch.Tensor
            预测的v风分量
        u_target : torch.Tensor
            目标的u风分量
        v_target : torch.Tensor
            目标的v风分量
        
        Returns
        -------
        loss : torch.Tensor
            涡度一致性损失（标量）
        """
        vorticity_pred = self.compute_vorticity(u_pred, v_pred)
        vorticity_target = self.compute_vorticity(u_target, v_target)
        
        # 计算涡度差的平方
        vorticity_diff_sq = torch.square(vorticity_pred - vorticity_target)
        # 球面积分
        loss = torch.sum(vorticity_diff_sq * self.quad_weights, dim=(-2, -1))
        return torch.mean(loss)
    
    def compute_vapor_transport_loss(self, u: torch.Tensor, v: torch.Tensor, 
                                     specific_humidity: torch.Tensor) -> torch.Tensor:
        """
        计算水汽输送损失
        
        水汽输送应该满足物理守恒定律
        
        Parameters
        ----------
        u : torch.Tensor
            u风分量，形状为 (batch, nlat, nlon)
        v : torch.Tensor
            v风分量，形状为 (batch, nlat, nlon)
        specific_humidity : torch.Tensor
            比湿，形状为 (batch, nlat, nlon)
        
        Returns
        -------
        loss : torch.Tensor
            水汽输送损失（标量）
        """
        # 计算水汽通量
        u_vapor = u * specific_humidity
        v_vapor = v * specific_humidity
        
        # 计算水汽通量的散度（应该接近0，表示守恒）
        vapor_divergence = self.compute_divergence(u_vapor, v_vapor)
        
        # 计算散度的平方
        vapor_div_sq = torch.square(vapor_divergence)
        # 球面积分
        loss = torch.sum(vapor_div_sq * self.quad_weights, dim=(-2, -1))
        return torch.mean(loss)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        out_variables: Optional[list] = None,
        variable_indices: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算物理约束损失
        
        Parameters
        ----------
        pred : torch.Tensor
            模型预测，形状为 (batch, n_vars, nlat, nlon)
        target : torch.Tensor
            真实值，形状为 (batch, n_vars, nlat, nlon)
        out_variables : list, optional
            输出变量名称列表，例如 ["2m_temperature", "10m_u_component_of_wind", ...]
        variable_indices : dict, optional
            变量索引字典，例如 {"u_wind": 1, "v_wind": 2, "specific_humidity": 6}
            如果为None，将根据out_variables自动推断
        
        Returns
        -------
        loss_dict : dict
            包含各项物理损失的字典
        """
        batch_size = pred.shape[0]
        
        # 默认变量索引（基于训练代码中的out_vars顺序）
        if variable_indices is None:
            if out_variables is not None:
                variable_indices = {}
                for i, var in enumerate(out_variables):
                    if "u_component_of_wind" in var.lower() or "u_wind" in var.lower():
                        variable_indices["u_wind"] = i
                    elif "v_component_of_wind" in var.lower() or "v_wind" in var.lower():
                        variable_indices["v_wind"] = i
                    elif "specific_humidity" in var.lower():
                        variable_indices["specific_humidity"] = i
            else:
                # 默认索引（基于训练代码）
                variable_indices = {
                    "u_wind": 1,  # "10m_u_component_of_wind"
                    "v_wind": 2,  # "10m_v_component_of_wind"
                    "specific_humidity": 6,  # "specific_humidity_850"
                }
        
        loss_dict = {}
        total_loss = 0.0
        
        # 提取风场分量
        if "u_wind" in variable_indices and "v_wind" in variable_indices:
            u_idx = variable_indices["u_wind"]
            v_idx = variable_indices["v_wind"]
            
            u_pred = pred[:, u_idx, :, :]  # (batch, nlat, nlon)
            v_pred = pred[:, v_idx, :, :]
            u_target = target[:, u_idx, :, :]
            v_target = target[:, v_idx, :, :]
            
            # 1. 散度约束损失
            if self.use_divergence:
                div_loss_pred = self.compute_divergence_loss(u_pred, v_pred)
                div_loss_target = self.compute_divergence_loss(u_target, v_target)
                # 使用目标散度作为参考（理想情况下应该接近0）
                divergence_loss = div_loss_pred * self.divergence_weight
                loss_dict["physics/divergence"] = divergence_loss
                total_loss += divergence_loss
            
            # 2. 涡度一致性损失
            if self.use_vorticity:
                vorticity_loss = self.compute_vorticity_consistency_loss(
                    u_pred, v_pred, u_target, v_target
                ) * self.vorticity_weight
                loss_dict["physics/vorticity"] = vorticity_loss
                total_loss += vorticity_loss
            
            # 3. 水汽输送损失
            if self.use_vapor and "specific_humidity" in variable_indices:
                q_idx = variable_indices["specific_humidity"]
                q_pred = pred[:, q_idx, :, :]
                q_target = target[:, q_idx, :, :]
                
                # 计算预测的水汽输送损失
                vapor_loss_pred = self.compute_vapor_transport_loss(u_pred, v_pred, q_pred)
                # 计算目标的水汽输送损失（作为参考）
                vapor_loss_target = self.compute_vapor_transport_loss(u_target, v_target, q_target)
                # 使用预测的损失（理想情况下应该接近目标）
                vapor_loss = vapor_loss_pred * self.vapor_weight
                loss_dict["physics/vapor_transport"] = vapor_loss
                total_loss += vapor_loss
        
        loss_dict["physics/total"] = total_loss
        
        return loss_dict


def create_physics_loss(
    nlat: int,
    nlon: int,
    grid: str = "equiangular",
    divergence_weight: float = 1.0,
    vorticity_weight: float = 0.5,
    vapor_weight: float = 0.5,
    use_divergence: bool = True,
    use_vorticity: bool = True,
    use_vapor: bool = True,
) -> PhysicsLossS2:
    """
    创建物理损失函数的便捷函数
    
    Parameters
    ----------
    nlat : int
        纬度方向的分辨率
    nlon : int
        经度方向的分辨率
    grid : str, optional
        网格类型，默认为 "equiangular"
    divergence_weight : float, optional
        散度损失的权重，默认为 1.0
    vorticity_weight : float, optional
        涡度损失的权重，默认为 0.5
    vapor_weight : float, optional
        水汽损失的权重，默认为 0.5
    use_divergence : bool, optional
        是否使用散度约束，默认为 True
    use_vorticity : bool, optional
        是否使用涡度约束，默认为 True
    use_vapor : bool, optional
        是否使用水汽约束，默认为 True
    
    Returns
    -------
    PhysicsLossS2
        物理损失函数实例
    """
    return PhysicsLossS2(
        nlat=nlat,
        nlon=nlon,
        grid=grid,
        divergence_weight=divergence_weight,
        vorticity_weight=vorticity_weight,
        vapor_weight=vapor_weight,
        use_divergence=use_divergence,
        use_vorticity=use_vorticity,
        use_vapor=use_vapor,
    )


# ============================================================================
# 物理损失权重调度器
# ============================================================================

class PhysicsWeightScheduler:
    """
    物理损失权重调度器
    
    支持多种权重调整策略：
    1. 线性增长：从初始权重线性增长到目标权重
    2. 阶梯增长：在特定epoch阶段增加权重
    3. 余弦退火：使用余弦函数平滑调整权重
    4. 自适应调整：根据验证损失自动调整权重
    """
    
    def __init__(
        self,
        initial_weight: float = 0.01,
        final_weight: float = 0.1,
        strategy: str = "linear",
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        step_epochs: Optional[list] = None,
        step_weights: Optional[list] = None,
    ):
        """
        初始化权重调度器
        
        Parameters
        ----------
        initial_weight : float
            初始权重（默认 0.01）
        final_weight : float
            最终权重（默认 0.1）
        strategy : str
            调度策略：'linear', 'step', 'cosine', 'adaptive'
        warmup_epochs : int
            预热epoch数（线性增长和余弦退火使用）
        total_epochs : int
            总epoch数
        step_epochs : list, optional
            阶梯增长的epoch列表（用于'step'策略）
        step_weights : list, optional
            阶梯增长的权重列表（用于'step'策略）
        """
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.step_epochs = step_epochs or []
        self.step_weights = step_weights or []
        self.current_weight = initial_weight
        
        # 验证参数
        if strategy == "step":
            if len(self.step_epochs) != len(self.step_weights):
                raise ValueError("step_epochs 和 step_weights 的长度必须相同")
            if len(self.step_epochs) == 0:
                raise ValueError("step策略需要提供step_epochs和step_weights")
    
    def get_weight(self, epoch: int) -> float:
        """
        根据当前epoch获取权重
        
        Parameters
        ----------
        epoch : int
            当前epoch（从0开始）
        
        Returns
        -------
        float
            当前epoch的权重
        """
        if self.strategy == "linear":
            return self._linear_schedule(epoch)
        elif self.strategy == "step":
            return self._step_schedule(epoch)
        elif self.strategy == "cosine":
            return self._cosine_schedule(epoch)
        elif self.strategy == "adaptive":
            # 自适应策略需要外部调用update_weight方法
            return self.current_weight
        else:
            raise ValueError(f"未知的策略: {self.strategy}")
    
    def _linear_schedule(self, epoch: int) -> float:
        """线性增长策略"""
        if epoch < self.warmup_epochs:
            # 线性增长阶段
            progress = epoch / self.warmup_epochs
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * progress
        else:
            # 保持最终权重
            weight = self.final_weight
        self.current_weight = weight
        return weight
    
    def _step_schedule(self, epoch: int) -> float:
        """阶梯增长策略"""
        weight = self.initial_weight
        for step_epoch, step_weight in zip(self.step_epochs, self.step_weights):
            if epoch >= step_epoch:
                weight = step_weight
        self.current_weight = weight
        return weight
    
    def _cosine_schedule(self, epoch: int) -> float:
        """余弦退火策略"""
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            progress = epoch / self.warmup_epochs
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * progress
        else:
            # 余弦退火阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            # 使用余弦函数从final_weight平滑变化
            weight = self.final_weight * (0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))))
            weight = weight.item()
        self.current_weight = weight
        return weight
    
    def update_weight(self, new_weight: float):
        """
        手动更新权重（用于自适应策略）
        
        Parameters
        ----------
        new_weight : float
            新的权重值
        """
        self.current_weight = new_weight
    
    def step(self, epoch: int):
        """
        更新权重（兼容PyTorch调度器接口）
        
        Parameters
        ----------
        epoch : int
            当前epoch
        """
        self.get_weight(epoch)


class AdaptivePhysicsWeightScheduler(PhysicsWeightScheduler):
    """
    自适应物理损失权重调度器
    
    根据验证损失自动调整权重：
    - 如果验证损失改善，逐渐增加物理损失权重
    - 如果验证损失恶化，保持或减少权重
    """
    
    def __init__(
        self,
        initial_weight: float = 0.01,
        max_weight: float = 0.5,
        min_weight: float = 0.001,
        improvement_threshold: float = 0.01,
        weight_increase_rate: float = 1.1,
        weight_decrease_rate: float = 0.9,
        patience: int = 5,
    ):
        """
        初始化自适应调度器
        
        Parameters
        ----------
        initial_weight : float
            初始权重
        max_weight : float
            最大权重
        min_weight : float
            最小权重
        improvement_threshold : float
            改善阈值（相对改善百分比）
        weight_increase_rate : float
            权重增加速率（每次乘以这个值）
        weight_decrease_rate : float
            权重减少速率（每次乘以这个值）
        patience : int
            容忍多少个epoch没有改善
        """
        super().__init__(initial_weight=initial_weight, strategy="adaptive")
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.improvement_threshold = improvement_threshold
        self.weight_increase_rate = weight_increase_rate
        self.weight_decrease_rate = weight_decrease_rate
        self.patience = patience
        
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
    
    def update(self, val_loss: float, epoch: int) -> float:
        """
        根据验证损失更新权重
        
        Parameters
        ----------
        val_loss : float
            当前验证损失
        epoch : int
            当前epoch
        
        Returns
        -------
        float
            更新后的权重
        """
        # 检查是否有改善
        improvement = (self.best_val_loss - val_loss) / self.best_val_loss if self.best_val_loss > 0 else 0
        
        if improvement > self.improvement_threshold:
            # 验证损失有改善，增加权重
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            new_weight = min(self.current_weight * self.weight_increase_rate, self.max_weight)
            self.current_weight = new_weight
        else:
            # 验证损失没有改善
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                # 超过容忍度，减少权重
                new_weight = max(self.current_weight * self.weight_decrease_rate, self.min_weight)
                self.current_weight = new_weight
                self.no_improvement_count = 0
        
        return self.current_weight


def create_physics_weight_scheduler(
    strategy: str = "linear",
    initial_weight: float = 0.01,
    final_weight: float = 0.1,
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    **kwargs
) -> PhysicsWeightScheduler:
    """
    创建物理损失权重调度器的便捷函数
    
    Parameters
    ----------
    strategy : str
        调度策略：'linear', 'step', 'cosine', 'adaptive'
    initial_weight : float
        初始权重
    final_weight : float
        最终权重
    warmup_epochs : int
        预热epoch数
    total_epochs : int
        总epoch数
    **kwargs
        其他参数（传递给调度器）
    
    Returns
    -------
    PhysicsWeightScheduler
        权重调度器实例
    
    Examples
    --------
    >>> # 线性增长策略
    >>> scheduler = create_physics_weight_scheduler(
    ...     strategy="linear",
    ...     initial_weight=0.01,
    ...     final_weight=0.1,
    ...     warmup_epochs=20,
    ...     total_epochs=100
    ... )
    
    >>> # 阶梯增长策略
    >>> scheduler = create_physics_weight_scheduler(
    ...     strategy="step",
    ...     initial_weight=0.01,
    ...     step_epochs=[10, 30, 50],
    ...     step_weights=[0.01, 0.05, 0.1],
    ...     total_epochs=100
    ... )
    
    >>> # 自适应策略
    >>> scheduler = AdaptivePhysicsWeightScheduler(
    ...     initial_weight=0.01,
    ...     max_weight=0.5,
    ...     patience=5
    ... )
    """
    if strategy == "adaptive":
        return AdaptivePhysicsWeightScheduler(
            initial_weight=initial_weight,
            max_weight=kwargs.get("max_weight", 0.5),
            min_weight=kwargs.get("min_weight", 0.001),
            **{k: v for k, v in kwargs.items() if k not in ["max_weight", "min_weight"]}
        )
    else:
        return PhysicsWeightScheduler(
            initial_weight=initial_weight,
            final_weight=final_weight,
            strategy=strategy,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            **kwargs
        )

