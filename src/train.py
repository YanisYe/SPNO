import os
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import logging
import numpy as np
import sys
import swanlab
from datetime import datetime
# 添加项目根目录和 src 目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
from .sp_operator import ClimaX
from .eval import evaluate, create_physics_loss_metric
from .datamodule import GlobalForecastDataModule as datamodule2
from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .spherical_encoder import WeatherSphericalEncoder
from .metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_mse_pde_loss_gradient,
    pearson,
)

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

model_name = "Spherical_Physics_Operator_Downscaling"
default_vars = ['land_sea_mask', 'orography', 'lattitude', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500', 'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850', 'relative_humidity_925', 'relative_humidity_1000', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000']
root_dir_low=r'/home/hunter/workspace/climate/mnt/'
root_dir_high=r'/home/hunter/workspace/climate/mnt/2.8125deg_npz/'

def ensure_directory_exists(directory_path):
    # 检查指定的路径是否存在
    if not os.path.exists(directory_path):
        # 如果不存在，创建新的文件夹
        os.makedirs(directory_path)
        print(f"文件夹'{directory_path}'已创建。")
    else:
        print(f"文件夹'{directory_path}'已存在。")

'''Loader Data'''
batch_size = 60
accumulation_steps = 1
pred_range = 1 #T
compete_len = 28000000
device = "cuda"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_root = os.path.join(base_dir, "logs", model_name)
num_epochs = 500 
stride_ = 2
H, W = 32, 64


out_vars = [
"2m_temperature",
"10m_u_component_of_wind",
"10m_v_component_of_wind",
"geopotential_500",
"temperature_850",
"relative_humidity_500",
"specific_humidity_850"
]

low_data = datamodule2(buffer_size=100,root_dir=root_dir_low,variables=default_vars,out_variables=out_vars,batch_size=batch_size,predict_range=pred_range, num_workers=1)
high_data = datamodule2(buffer_size=100,root_dir=root_dir_high,variables=out_vars,out_variables=out_vars,batch_size=batch_size,predict_range=pred_range, num_workers=1)

high_data.setup()
low_data.setup()
low_lat, low_lon = low_data.get_lat_lon()
lat, lon = high_data.get_lat_lon()
lat_high, lon_high = lat.shape[0], lon.shape[0]
val_clim = high_data.val_clim
test_clim = high_data.test_clim
normalization = high_data.transforms
mean_norm, std_norm = normalization.mean, normalization.std
mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
denormalization = transforms.Normalize(mean_denorm, std_denorm)

theta_cutoff = None

low_train_dataloader = low_data.train_dataloader()
low_test_dataloader = low_data.test_dataloader()
low_val_dataloader = low_data.val_dataloader()
# low_train_dataloader2 = low_data2.train_dataloader()

high_train_dataloader = high_data.train_dataloader()
high_test_dataloader = high_data.test_dataloader()
high_val_dataloader = high_data.val_dataloader()
# high_train_dataloader2 = high_data2.train_dataloader()

low_lat = torch.tensor(low_lat, dtype=torch.float32)
low_lon = torch.tensor(low_lon, dtype=torch.float32)
xx_low, yy_low = torch.meshgrid(low_lat, low_lon, indexing="ij")
coords = torch.hstack([xx_low.flatten()[:, None], yy_low.flatten()[:, None]]).to(device)

num_scales = 4

def train(
    encoder, 
    model: ClimaX, 
    epoch, 
    high_dataloader, 
    low_dataloader, 
    coords, 
    optimizer, 
    scheduler, 
    device, 
    accumulation_steps=6, 
    global_step=0, 
    physics_loss=None, 
    physics_weight=0.0, 
    physics_loss_normalize=False
    ):

    model.train()
    encoder.train()  # 确保 encoder 也处于训练模式
    total_loss = 0
    cnt = 0
    start_time = time.time()  # 开始时间记录
    current_step = global_step  # 使用全局 step 计数器
    
    # 用于归一化物理损失的移动平均（EMA）
    if physics_loss_normalize:
        if not hasattr(train, 'mse_ema'):
            train.mse_ema = None
        if not hasattr(train, 'physics_ema'):
            train.physics_ema = None
        ema_decay = 0.99  # 指数移动平均衰减率

    
    
    for batch_idx, (high_batch, low_batch) in enumerate(zip(high_dataloader, low_dataloader)):
        batch_start_time = time.time()  # 记录每个 batch 的开始时间
        cnt += 1
        optimizer.zero_grad()# 清空梯度

        x,_,_, variables,_ = low_batch
        y,_,_, _, out_variables = high_batch
        # x = x.to(device)
        x = x.squeeze(1).to(device)
        y = y.squeeze(1).to(device)


        # 应用同步旋转增强 todo
        state = encoder(x).to(device)
        
        # y = y.to(device)
        
        batch_size, _, high_h, high_w = y.shape
        res = F.interpolate(x[:,torch.tensor([3, 4, 5, 8, 35, 40, 51]).type(torch.long).to(x.device),:], 
                           size=(high_h, high_w), mode='bicubic', align_corners=False).to(device)
        
        # 计算 attention scale（动态调整）
        # attn_scale =  min(min(1e-2, epoch / 50 * 1e-2), 1e-1) #todo: 换成cosine annealing
        # attn_scale = 1e-2

        # 前向传播和计算损失
        loss_dict, pred_output = model.forward(state, x , y, res, out_variables, [lat_weighted_mse], lat=lat)
        loss_dict = loss_dict[0]
        mse_loss = loss_dict["loss"] / accumulation_steps  # 除以累计步数，平摊损失
        
        # 添加物理约束损失（如果启用）
        physics_loss_value = None
        physics_loss_dict = None
        if physics_loss is not None and physics_weight > 0:
            # pred_output 是模型的预测输出，形状为 (batch, n_vars, nlat, nlon)
            # y 是真实值，形状为 (batch, n_vars, nlat, nlon)
            physics_loss_dict = physics_loss(pred_output, y, out_variables=out_variables)
            physics_loss_raw = physics_loss_dict["physics/total"]
            
            # 归一化物理损失（使其与MSE损失尺度匹配）
            if physics_loss_normalize:
                mse_loss_value = loss_dict["loss"].detach()  # MSE损失值（仅用于记录）
                physics_loss_value_detached = physics_loss_raw.detach()  # 仅用于记录
                
                # 更新移动平均
                if train.mse_ema is None:
                    train.mse_ema = mse_loss_value.item()
                else:
                    train.mse_ema = ema_decay * train.mse_ema + (1 - ema_decay) * mse_loss_value.item()
                
                if train.physics_ema is None:
                    train.physics_ema = physics_loss_value_detached.item()
                else:
                    train.physics_ema = ema_decay * train.physics_ema + (1 - ema_decay) * physics_loss_value_detached.item()
                
                # 归一化：使物理损失的尺度与MSE损失匹配
                if train.physics_ema > 1e-8:  # 避免除零
                    normalization_factor = train.mse_ema / train.physics_ema
                    physics_loss_normalized = physics_loss_raw * normalization_factor
                    loss_dict["physics/normalization_factor"] = normalization_factor
                else:
                    physics_loss_normalized = physics_loss_raw
                    loss_dict["physics/normalization_factor"] = 1.0
            else:
                physics_loss_normalized = physics_loss_raw
            
            physics_loss_value = physics_loss_normalized * physics_weight / accumulation_steps
            
            # 将物理损失添加到loss_dict中以便记录（使用detach的值用于记录）
            for k, v in physics_loss_dict.items():
                loss_dict[f"physics/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
            loss_dict["physics/total_normalized"] = physics_loss_normalized.item() if isinstance(physics_loss_normalized, torch.Tensor) else physics_loss_normalized

        # 标准反向传播
        loss = mse_loss
        if physics_loss_value is not None:
            loss = loss + physics_loss_value
        
        # 反向传播
        loss.backward()

        # 每隔 accumulation_steps 更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新参数
            scheduler.step()
            optimizer.zero_grad()  # 清空梯度

            batch_time = time.time() - batch_start_time  # 计算每个 batch 的时间
            print(f"Epoch [{epoch}], Batch [{batch_idx+1}], Train Loss: {loss_dict}, Batch Time: {batch_time:.2f} s")
            logging.info(f"Epoch [{epoch}], Batch [{batch_idx+1}], Train Loss: {loss_dict}, Batch Time: {batch_time:.2f} s")
            
            # 记录到 swanlab
            current_step += 1
            # 解包 loss_dict 中的所有指标
            swanlab_log_dict = {
                "train/loss": loss_dict["loss"].item(),
                "train/batch_time": batch_time,
                "train/learning_rate": scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr'],
            }
            # 添加 loss_dict 中的所有指标
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    swanlab_log_dict[f"train/{k}"] = v.item() if v.numel() == 1 else v.cpu().numpy()
                elif isinstance(v, (int, float)):
                    swanlab_log_dict[f"train/{k}"] = v
                elif isinstance(v, np.ndarray):
                    swanlab_log_dict[f"train/{k}"] = v.item() if v.size == 1 else v
             
            swanlab.log(swanlab_log_dict, step=current_step)

        total_loss += loss.item() * accumulation_steps  # 累加未经平摊的损失值

    # 处理未被整除的批次
    if cnt % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    epoch_time = time.time() - start_time  # 计算整个 epoch 的时间
    avg_loss = total_loss / cnt if cnt > 0 else 0

    return avg_loss, epoch_time, current_step

# 测试/验证循环

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer, scheduler, encoder=None):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if encoder is not None and 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_test_loss = checkpoint.get('best_test_loss', float('inf'))
        best_val_epoch = checkpoint['best_val_epoch']
        best_test_epoch = checkpoint.get('best_test_epoch', 0)
        return epoch - 1, best_val_loss, best_test_loss, best_val_epoch, best_test_epoch
    else:
        return 0, float('inf'), float('inf'), 0, 0

def generate_coord(lat, lon):
   
    lat = torch.tensor(lat, dtype=torch.float32)
    lon = torch.tensor(lon, dtype=torch.float32)
    
    # Calculate latitude weights for high resolution coordinates
    w_lat = np.cos(np.deg2rad(lat.numpy()))
    w_lat = w_lat / w_lat.mean()  # Normalize
    w_lat = torch.from_numpy(w_lat).to(dtype=lat.dtype, device=lat.device)
    
    # Apply latitude weights to low and high latitude tensors
    weighted_lat = lat * w_lat
    
    # Create meshgrid for low and high resolution coordinate

    return weighted_lat, lon

def main():
    # 创建带时间戳的日志文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_folder = os.path.join(save_root, str(pred_range), f"coords_{timestamp}")
    ensure_directory_exists(logging_folder)
    print(f"日志和checkpoint将保存到: {logging_folder}")
    
    lr = 5e-4
    betas = (0.9, 0.95)
    pde_weight = 0.01
    fourier_weight = 1.0
    # 物理损失参数（可选）
    use_physics_loss = True  # 设置为 True 启用物理损失
    physics_weight = 0.01  # 物理损失的权重（如果使用调度器，这是初始权重）
    physics_weight_schedule = True  # 是否使用权重调度器
    physics_weight_strategy = "adaptive"  # 调度策略: "linear", "step", "cosine", "adaptive"
    physics_weight_initial = 0.0001  # 初始权重
    physics_weight_final = 0.01  # 最终权重
    physics_weight_warmup_epochs = 50  # 预热epoch数
    physics_loss_normalize = True  # 是否归一化物理损失
    physics_divergence_weight = 1.0
    physics_vorticity_weight = 0.5
    physics_vapor_weight = 0.5
    # 新增：针对geopotential_500和temperature_850的物理约束权重
    physics_geopotential_weight = 1.0  # 位势高度梯度约束权重
    physics_temperature_weight = 1.0  # 温度平滑性约束权重
    use_geopotential_constraint = True  # 启用位势高度约束
    use_temperature_constraint = True   # 启用温度约束
    use_spherical_attn = True
    use_spherical_conv = True
    # 早停机制参数
    early_stopping_patience = 30000  # 容忍多少个epoch没有改善
    early_stopping_min_delta = 1e-6  # 最小改善阈值
    # global operator token
    num_global_operator_token = 16
    # 初始化物理损失（如果启用）
    physics_loss_fn = None
    physics_weight_scheduler = None
    if use_physics_loss:
        from phys_loss_v2 import create_physics_loss, create_physics_weight_scheduler
       
        physics_loss_fn = create_physics_loss(
            nlat=lat_high, 
            nlon = lon_high,
            grid="equiangular",     
            divergence_weight=physics_divergence_weight,
            vorticity_weight=physics_vorticity_weight,
            vapor_weight=physics_vapor_weight,
            use_divergence=True,
            use_vorticity=True,
            use_vapor=True,
            # geopotential_weight=physics_geopotential_weight,
            # temperature_weight=physics_temperature_weight,
            # use_geopotential=use_geopotential_constraint,
            # use_temperature=use_temperature_constraint,
        ).to(device)
        
        # 初始化权重调度器
        if physics_weight_schedule:
            physics_weight_scheduler = create_physics_weight_scheduler(
                strategy=physics_weight_strategy,
                initial_weight=physics_weight_initial,
                final_weight=physics_weight_final,
                warmup_epochs=physics_weight_warmup_epochs,
                total_epochs=num_epochs,
            )
            print(f"✓ 物理损失已启用，使用权重调度器: {physics_weight_strategy}")
            print(f"  初始权重: {physics_weight_initial}, 最终权重: {physics_weight_final}")
            print(f"  预热epoch数: {physics_weight_warmup_epochs}")
            print(f"  损失归一化: {physics_loss_normalize}")
            logging.info(f"✓ 物理损失已启用，使用权重调度器: {physics_weight_strategy}")
            logging.info(f"  初始权重: {physics_weight_initial}, 最终权重: {physics_weight_final}")
            logging.info(f"  预热epoch数: {physics_weight_warmup_epochs}")
            logging.info(f"  损失归一化: {physics_loss_normalize}")
        else:
            print(f"✓ 物理损失已启用，固定权重: {physics_weight}")
            logging.info(f"✓ 物理损失已启用，固定权重: {physics_weight}")
    else:
        print("物理损失未启用")
        logging.info("物理损失未启用")
    
    # 初始化 swanlab
    swanlab.init(
        project="climate_downscaling",
        experiment_name=f"lr={lr}_pred_range={pred_range}_pdeweight={pde_weight}_coords_Spherical_Operator_with_physics_loss_v11_use_spherical_conv_False",
        config={
            "lr": lr,
            "betas": betas,
            "pde_weight": pde_weight,
            "fourier_weight": fourier_weight,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "pred_range": pred_range,
            "num_epochs": num_epochs,
            "device": device,
            "model_name": model_name,
            "out_vars": out_vars,
            "use_physics_loss": use_physics_loss,
            "physics_weight": physics_weight,
            "physics_weight_schedule": physics_weight_schedule,
            "physics_weight_strategy": physics_weight_strategy,
            "physics_weight_initial": physics_weight_initial,
            "physics_weight_final": physics_weight_final,
            "physics_weight_warmup_epochs": physics_weight_warmup_epochs,
            "physics_divergence_weight": physics_divergence_weight,
            "physics_vorticity_weight": physics_vorticity_weight,
            "physics_vapor_weight": physics_vapor_weight,
            "physics_geopotential_weight": physics_geopotential_weight,
            "physics_temperature_weight": physics_temperature_weight,
            "use_geopotential_constraint": use_geopotential_constraint,
            "use_temperature_constraint": use_temperature_constraint,
            "physics_loss_normalize": physics_loss_normalize,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        }
    )
    
    logging.basicConfig(filename=os.path.join(logging_folder, "2times_coords,lr={}_pred_range={}_pdeweight={}.log".format(lr, pred_range, pde_weight)), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    
    jj, kk = generate_coord(low_lat, low_lon)
    jjj, kkk = generate_coord(lat, lon)

    # Generate latitude weights for encoder (cosine of latitude)
    # low_lat is already a tensor from line 125
    lat_weights = torch.cos(torch.deg2rad(low_lat))
    
    encoder = WeatherSphericalEncoder(
        in_channels=len(default_vars),
        lat_weights=lat_weights,
        num_scales=num_scales,
        use_spherical_conv=use_spherical_conv,
    )
    model = ClimaX(default_vars=default_vars,out_dim = len(out_vars),low_gird=[jj, kk], high_gird=[jjj, kkk], num_global_operator_token=num_global_operator_token, theta_cutoff=theta_cutoff, dec_num_heads=4, use_spherical_attn=use_spherical_attn)
    

    
    ipe = (2*(1460))/(batch_size * accumulation_steps)

    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=1e-5)
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5*ipe, eta_min=1e-8, max_epochs=num_epochs * ipe,warmup_start_lr=1e-8)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1 *ipe, eta_min=1e-8, max_epochs=num_epochs*ipe, warmup_start_lr=1e-8)
    model = model.to(device)
    encoder = encoder.to(device)
    start_epoch = 0
    # # 尝试加载检查点
    # start_epoch, best_val_loss, best_test_loss, best_val_epoch, best_test_epoch = load_checkpoint(f"{logging_folder}/our_bicubic_epoch=150,no_chazhi_climax_FFeatrue_kenel_all.pth.tar", model, optimizer, scheduler, encoder)
    best_val_loss = float('inf')
    best_val_w_rmse = float('inf')  # 单独跟踪 w_rmse 的最佳值
    best_test_loss = float('inf')
    best_val_epoch = 0
    best_val_w_rmse_epoch = 0  # w_rmse 最佳模型的 epoch
    best_test_epoch = 0
    s = torch.tensor([2], dtype=torch.int).to(device)
    global_step = 0  # 初始化全局 step 计数器
    # 早停机制变量
    patience_counter = 0  # 记录没有改善的epoch数
    best_model_state = None  # 保存最佳模型状态（基于主要指标）
    best_w_rmse_model_state = None  # 保存 w_rmse 最佳模型状态
    
    for epoch in range(start_epoch, num_epochs):
        # 更新物理损失权重（如果使用调度器）
        current_physics_weight = physics_weight
        if use_physics_loss and physics_weight_scheduler is not None:
            if physics_weight_strategy == "adaptive":
                # 自适应策略需要先获取验证损失，在验证后更新
                pass  # 将在验证后更新
            else:
                current_physics_weight = physics_weight_scheduler.get_weight(epoch)
        
        # train_loss = train(encoder, model, epoch, high_train_dataloader,low_train_dataloader, coords, optimizer,scheduler, device, accumulation_steps)
        train_loss, train_time, global_step = train(encoder, model, epoch, high_train_dataloader,low_train_dataloader, coords, optimizer,scheduler, device, accumulation_steps, global_step, physics_loss=physics_loss_fn, physics_weight=current_physics_weight, physics_loss_normalize=physics_loss_normalize)
        val_loss = evaluate(encoder, model, low_val_dataloader, high_val_dataloader, device,
                            denormalization=denormalization, lat=lat, clim=val_clim, 
                            pred_range=pred_range, physics_loss_fn=physics_loss_fn)
        test_loss = evaluate(encoder, model, low_test_dataloader, high_test_dataloader, device,
                            denormalization=denormalization, lat=lat, clim=test_clim, 
                            pred_range=pred_range, physics_loss_fn=physics_loss_fn)
        
        # 自适应策略：根据验证损失更新权重
        if use_physics_loss and physics_weight_scheduler is not None and physics_weight_strategy == "adaptive":
            val_loss_key = None
            if 'loss' in val_loss:
                val_loss_key = 'loss'
            elif 'w_rmse' in val_loss:
                val_loss_key = 'w_rmse'
            else:
                for k, v in val_loss.items():
                    if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                        val_loss_key = k
                        break
            
            if val_loss_key is not None:
                current_val_loss = val_loss[val_loss_key]
                if isinstance(current_val_loss, torch.Tensor):
                    current_val_loss = current_val_loss.item()
                current_physics_weight = physics_weight_scheduler.update(current_val_loss, epoch)
                print(f"  自适应物理损失权重更新为: {current_physics_weight:.6f}")
                logging.info(f"  自适应物理损失权重更新为: {current_physics_weight:.6f}")

        # 记录 epoch 级别的指标到 swanlab
        epoch_log = {
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "train/epoch_time": train_time,
        }
        
        # 记录物理损失权重
        if use_physics_loss:
            epoch_log["train/physics_weight"] = current_physics_weight
        
        # 添加验证集指标
        for k, v in val_loss.items():
            if isinstance(v, torch.Tensor):
                epoch_log[f"val/{k}"] = v.item() if v.numel() == 1 else v.cpu().numpy()
            elif isinstance(v, (int, float)):
                epoch_log[f"val/{k}"] = v
            elif isinstance(v, np.ndarray):
                epoch_log[f"val/{k}"] = v.item() if v.size == 1 else v
        
        # 添加测试集指标
        for k, v in test_loss.items():
            if isinstance(v, torch.Tensor):
                epoch_log[f"test/{k}"] = v.item() if v.numel() == 1 else v.cpu().numpy()
            elif isinstance(v, (int, float)):
                epoch_log[f"test/{k}"] = v
            elif isinstance(v, np.ndarray):
                epoch_log[f"test/{k}"] = v.item() if v.size == 1 else v
        
        swanlab.log(epoch_log, step=epoch + 1)

        # 早停机制：选择验证损失的主要指标（优先使用 'loss'，否则使用 'w_rmse' 或第一个可用指标）
        val_loss_key = None
        if 'loss' in val_loss:
            val_loss_key = 'loss'
        elif 'w_rmse' in val_loss:
            val_loss_key = 'w_rmse'
        else:
            # 使用第一个可用的数值指标
            for k, v in val_loss.items():
                if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                    val_loss_key = k
                    break
        
        # 同时跟踪 'loss' 和 'w_rmse' 的最佳模型
        if val_loss_key is not None:
            current_val_loss = val_loss[val_loss_key]
            if isinstance(current_val_loss, torch.Tensor):
                current_val_loss = current_val_loss.item()
            
            # 检查主要指标是否有改善（用于早停）
            if current_val_loss < best_val_loss - early_stopping_min_delta:
                # 验证损失有改善
                best_val_loss = current_val_loss
                best_val_epoch = epoch
                patience_counter = 0
                # 保存最佳模型状态（基于主要指标）
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                }
                print(f"✓ 验证损失改善: {current_val_loss:.6f} (最佳: {best_val_loss:.6f})")
                logging.info(f"✓ 验证损失改善: {current_val_loss:.6f} (最佳: {best_val_loss:.6f})")
            else:
                # 验证损失没有改善
                patience_counter += 1
                print(f"⚠ 验证损失未改善 ({patience_counter}/{early_stopping_patience}): {current_val_loss:.6f} (最佳: {best_val_loss:.6f})")
                logging.info(f"⚠ 验证损失未改善 ({patience_counter}/{early_stopping_patience}): {current_val_loss:.6f} (最佳: {best_val_loss:.6f})")
                
                # 检查是否触发早停
                if patience_counter >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"早停触发！在 Epoch {epoch+1} 停止训练")
                    print(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_val_epoch+1})")
                    print(f"最佳 w_rmse: {best_val_w_rmse:.6f} (Epoch {best_val_w_rmse_epoch+1})")
                    print(f"{'='*60}")
                    logging.info(f"早停触发！在 Epoch {epoch+1} 停止训练")
                    logging.info(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_val_epoch+1})")
                    logging.info(f"最佳 w_rmse: {best_val_w_rmse:.6f} (Epoch {best_val_w_rmse_epoch+1})")
                    
                    # 恢复最佳模型（基于主要指标）
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state['model_state_dict'])
                        encoder.load_state_dict(best_model_state['encoder_state_dict'])
                        print("已恢复最佳模型状态")
                        logging.info("已恢复最佳模型状态")
                    
                    # 保存最佳模型检查点（基于主要指标）
                    if best_model_state is not None:
                        checkpoint_path = os.path.join(logging_folder, f"{model_name}_best_epoch{best_val_epoch+1}.pth.tar")
                        save_checkpoint(best_model_state, checkpoint_path)
                        print(f"已保存最佳模型到: {checkpoint_path}")
                        logging.info(f"已保存最佳模型到: {checkpoint_path}")
                    
                    # 保存 w_rmse 最佳模型检查点
                    if best_w_rmse_model_state is not None:
                        checkpoint_path_w_rmse = os.path.join(logging_folder, f"{model_name}_best_w_rmse_epoch{best_val_w_rmse_epoch+1}.pth.tar")
                        save_checkpoint(best_w_rmse_model_state, checkpoint_path_w_rmse)
                        print(f"已保存最佳 w_rmse 模型到: {checkpoint_path_w_rmse}")
                        logging.info(f"已保存最佳 w_rmse 模型到: {checkpoint_path_w_rmse}")
                    
                    break
        
        # 单独跟踪和保存 w_rmse 最佳模型
        if 'w_rmse' in val_loss:
            current_w_rmse = val_loss['w_rmse']
            if isinstance(current_w_rmse, torch.Tensor):
                current_w_rmse = current_w_rmse.item()
            
            # 检查 w_rmse 是否有改善
            if current_w_rmse < best_val_w_rmse - early_stopping_min_delta:
                best_val_w_rmse = current_w_rmse
                best_val_w_rmse_epoch = epoch
                # 保存 w_rmse 最佳模型状态
                best_w_rmse_model_state = {
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_w_rmse': best_val_w_rmse,
                }
                print(f"✓ w_rmse 改善: {current_w_rmse:.6f} (最佳: {best_val_w_rmse:.6f})")
                logging.info(f"✓ w_rmse 改善: {current_w_rmse:.6f} (最佳: {best_val_w_rmse:.6f})")
                
                # 立即保存 w_rmse 最佳模型
                checkpoint_path_w_rmse = os.path.join(logging_folder, f"{model_name}_best_w_rmse_epoch{epoch+1}.pth.tar")
                save_checkpoint(best_w_rmse_model_state, checkpoint_path_w_rmse)
                print(f"已保存最佳 w_rmse 模型到: {checkpoint_path_w_rmse}")
                logging.info(f"已保存最佳 w_rmse 模型到: {checkpoint_path_w_rmse}")
        
        # 保存最小验证集损失对应的模型（每10个epoch保存一次检查点）
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(logging_folder, f"{model_name}_epoch{epoch+1}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_w_rmse': best_val_w_rmse,
                'best_test_loss': best_test_loss,
                'best_val_epoch': best_val_epoch,
                'best_val_w_rmse_epoch': best_val_w_rmse_epoch,
                'best_test_epoch': best_test_epoch
            }, checkpoint_path)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss}, Test Loss: {test_loss}")
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss}, Test Loss: {test_loss}")
    
    # 训练结束后的处理
    if patience_counter < early_stopping_patience:
        print(f"\n训练完成！共训练了 {num_epochs} 个epoch")
        logging.info(f"训练完成！共训练了 {num_epochs} 个epoch")
        # 保存最终模型（基于主要指标）
        if best_model_state is not None:
            checkpoint_path = os.path.join(logging_folder, f"{model_name}_final.pth.tar")
            save_checkpoint(best_model_state, checkpoint_path)
            print(f"已保存最终模型到: {checkpoint_path}")
            logging.info(f"已保存最终模型到: {checkpoint_path}")
        
        # 保存 w_rmse 最佳模型（如果存在）
        if best_w_rmse_model_state is not None:
            checkpoint_path_w_rmse = os.path.join(logging_folder, f"{model_name}_final_w_rmse.pth.tar")
            save_checkpoint(best_w_rmse_model_state, checkpoint_path_w_rmse)
            print(f"已保存最终 w_rmse 最佳模型到: {checkpoint_path_w_rmse}")
            logging.info(f"已保存最终 w_rmse 最佳模型到: {checkpoint_path_w_rmse}")
        
        print(f"\n最佳模型总结:")
        print(f"  主要指标最佳: {best_val_loss:.6f} (Epoch {best_val_epoch+1})")
        if best_val_w_rmse < float('inf'):
            print(f"  w_rmse 最佳: {best_val_w_rmse:.6f} (Epoch {best_val_w_rmse_epoch+1})")
        logging.info(f"最佳模型总结: 主要指标={best_val_loss:.6f} (Epoch {best_val_epoch+1}), w_rmse={best_val_w_rmse:.6f} (Epoch {best_val_w_rmse_epoch+1})")

if __name__ == '__main__':
# 模型训练
    main()