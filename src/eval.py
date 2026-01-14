"""
推理脚本：用于测试保存的checkpoint的推理能力
参考 train.py 的结构
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from datetime import datetime
from torchvision.transforms import transforms
import argparse

# 添加项目根目录和 src 目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.sp_operator import ClimaX
from src.datamodule import GlobalForecastDataModule as datamodule2
from src.metrics import (
    lat_weighted_acc,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    pearson,
)
from src.spherical_encoder import WeatherSphericalEncoder
from src.phys_loss import create_physics_loss
from functools import partial


def generate_coord(lat, lon):
    """生成坐标（与train.py保持一致）"""
    lat = torch.tensor(lat, dtype=torch.float32)
    lon = torch.tensor(lon, dtype=torch.float32)
    
    # Calculate latitude weights for high resolution coordinates
    w_lat = np.cos(np.deg2rad(lat.numpy()))
    w_lat = w_lat / w_lat.mean()  # Normalize
    w_lat = torch.from_numpy(w_lat).to(dtype=lat.dtype, device=lat.device)
    
    # Apply latitude weights to low and high latitude tensors
    weighted_lat = lat * w_lat
    
    return weighted_lat, lon


def load_checkpoint_for_inference(filepath, model, encoder, device):
    """
    加载checkpoint用于推理（不需要optimizer和scheduler）
    
    Args:
        filepath: checkpoint文件路径
        model: 模型实例
        encoder: encoder实例
        device: 设备
    
    Returns:
        checkpoint信息字典
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果没有model_state_dict，尝试直接加载
        model.load_state_dict(checkpoint)
    
    # 加载encoder状态
    if encoder is not None:
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            print("警告: checkpoint中没有encoder_state_dict，跳过encoder加载")
    
    # 提取checkpoint信息
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'best_val_w_rmse': checkpoint.get('best_val_w_rmse', 'unknown'),
        'best_val_epoch': checkpoint.get('best_val_epoch', 'unknown'),
        'best_val_w_rmse_epoch': checkpoint.get('best_val_w_rmse_epoch', 'unknown'),
    }
    
    return checkpoint_info

def create_physics_loss_metric(physics_loss_fn):
    """
    创建物理损失 metric 函数
    
    Args:
        physics_loss_fn: 物理损失函数
    
    Returns:
        metric 函数，签名与其他 metric 函数相同
    """
    def physics_loss_metric(pred, y, transform, out_vars, lat, clim, log_postfix, inp_vars=None):
        """
        物理损失 metric
        
        Args:
            pred: [B, V, H, W] 预测值
            y: [B, V, H, W] 真实值
            transform: 变换函数（未使用，但保持接口一致）
            out_vars: 变量列表
            lat: 纬度数组
            clim: 气候态（未使用）
            log_postfix: 日志后缀
            inp_vars: 输入变量（未使用）
        
        Returns:
            包含物理损失的字典
        """
        if physics_loss_fn is None:
            return {}
        
        # 计算物理损失
        physics_loss_dict = physics_loss_fn(
            pred,
            y,
            out_variables=out_vars
        )
        
        # 转换为标量字典
        loss_dict = {}
        for k, v in physics_loss_dict.items():
            # k 可能已经包含 "physics/" 前缀，检查并处理
            if k.startswith("physics/"):
                key = f"{k}_{log_postfix}"
            else:
                key = f"physics/{k}_{log_postfix}"
            
            if isinstance(v, torch.Tensor):
                loss_dict[key] = v.item() if v.numel() == 1 else v.mean().item()
            else:
                loss_dict[key] = v
        
        return loss_dict
    
    return physics_loss_metric



def evaluate(encoder, model:ClimaX, low_dataloader, high_dataloader, device, 
             denormalization=None, lat=None, clim=None, pred_range=1, physics_loss_fn=None):
    """
    评估函数：对数据进行评估并计算指标
    
    Args:
        encoder: 编码器
        model: 模型
        low_dataloader: 低分辨率数据加载器
        high_dataloader: 高分辨率数据加载器
        device: 设备
        denormalization: 反归一化变换（如果为None，使用全局变量）
        lat: 纬度数组（如果为None，使用全局变量）
        clim: 气候态（如果为None，使用全局变量）
        pred_range: 预测范围（小时）
        physics_loss_fn: 物理损失函数（可选）
    
    Returns:
        指标字典
    """
    model.eval()
    encoder.eval()  # encoder 处于评估模式
    total_loss = 0
    all_loss_dicts = []
    total_batches = 0
    
    # 使用传入的参数，如果没有则使用全局变量（向后兼容）
    if denormalization is None:
        denormalization = globals().get('denormalization')
    if lat is None:
        lat = globals().get('lat')
    if clim is None:
        clim = globals().get('test_clim')
    
    # 创建物理损失 metric（如果提供了物理损失函数）
    metrics_list = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]
    if physics_loss_fn is not None:
        physics_metric = create_physics_loss_metric(physics_loss_fn)
        metrics_list.append(physics_metric)
    
    with torch.no_grad():
        cnt = 0
        for batch_idx, (high_batch, low_batch) in enumerate(zip(high_dataloader, low_dataloader)):
            total_batches += 1
            cnt += 1
            y, _, _, out_variables, _ = high_batch
            x,_,lead_times, variables, _ = low_batch
            x = x.squeeze(1).to(device)

            state = encoder(x)
            y = y.squeeze(1).to(device)

            # 计算 res：从低分辨率输入中选择某些变量并插值到高分辨率
            batch_size, _, high_h, high_w = y.shape
            res = F.interpolate(x[:,torch.tensor([3, 4, 5, 8, 35, 40, 51]).type(torch.long).to(x.device),:], 
                               size=(high_h, high_w), mode='bicubic', align_corners=False).to(device)

            # res = F.interpolate(x[:,torch.tensor([3, 4, 5]).type(torch.long).to(x.device),:], 
            #                    size=(high_h, high_w), mode='bicubic', align_corners=False).to(device)
           
            if pred_range < 24:
                log_postfix = f"{pred_range}_hours"
            else:
                days = int(pred_range / 24)
                log_postfix = f"{days}_days"

            loss_dict = model.evaluate(
                state,
                x,
                y,
                res,
                out_variables,
                transform=denormalization,
                metrics=metrics_list,
                lat=lat,
                clim=clim,
                log_postfix=log_postfix,
            )
            if batch_idx % 250 == 0: print(batch_idx)
            all_loss_dicts.append(loss_dict)

    # 初始化合并后的损失字典
    loss_dict_combined = {}
    for dd in all_loss_dicts:
        for d in dd:
            for k in d.keys():
                if k in loss_dict_combined:
                    loss_dict_combined[k] += d[k]
                else:
                    loss_dict_combined[k] = d[k]

    # 计算平均值
    loss_dict_avg = {k: v / total_batches for k, v in loss_dict_combined.items()}
    
    return loss_dict_avg

def inference(
    encoder,
    model: ClimaX,
    low_dataloader,
    high_dataloader,
    device,
    denormalization,
    lat,
    test_clim=None,
    pred_range=1,
    physics_loss_fn=None,
):
    """
    推理函数：对数据进行推理并计算指标
    
    Args:
        encoder: 编码器
        model: 模型
        low_dataloader: 低分辨率数据加载器
        high_dataloader: 高分辨率数据加载器
        device: 设备
        denormalization: 反归一化变换
        lat: 纬度数组
        test_clim: 测试集气候态（可选）
        pred_range: 预测范围（小时）
        save_predictions: 是否保存预测结果
        output_dir: 输出目录
        max_batches: 最大批次数（None表示处理所有批次）
        physics_loss_fn: 物理损失函数（可选）
    
    Returns:
        指标字典
    """
    # 创建虚拟坐标（train.evaluate 需要 coords 参数，但实际不使用）
    
    # 使用 train.py 中的 evaluate 函数
    loss_dict_avg = evaluate(
        encoder=encoder,
        model=model,
        low_dataloader=low_dataloader,
        high_dataloader=high_dataloader,
        device=device,
        denormalization=denormalization,
        lat=lat,
        clim=test_clim,
        pred_range=pred_range,
        physics_loss_fn=physics_loss_fn,
    )
    
    return loss_dict_avg


def main():
    parser = argparse.ArgumentParser(description="推理脚本：测试checkpoint的推理能力")
    parser.add_argument(
        "--checkpoint",
        type=str,
        # required=True,
        default="logs/SphereConv_Physics_Downscaling_Spherical_Operator_with_physics_loss_v11/1/coords_20260105_140420/SphereConv_Physics_Downscaling_Spherical_Operator_no_physics_loss_v11_final_w_rmse.pth.tar",
        help="checkpoint文件路径"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="使用的数据集分割（train/val/test）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda/cpu）"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="最大批次数（None表示处理所有批次）"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="是否保存预测结果"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（如果保存预测结果）"
    )
    parser.add_argument(
        "--low_res_root",
        type=str,
        default="/home/hunter/workspace/climate/mnt/",
        help="低分辨率数据根目录"
    )
    parser.add_argument(
        "--high_res_root",
        type=str,
        default="/home/hunter/workspace/climate/mnt/2.8125deg_npz/",
        help="高分辨率数据根目录"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "inference",
            f"{checkpoint_name}_{args.split}_{timestamp}"
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, "inference.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.info(f"开始推理，checkpoint: {args.checkpoint}")
    logging.info(f"数据集分割: {args.split}")
    logging.info(f"设备: {device}")
    
    # 定义变量（与train_v11.py保持一致）
    default_vars = [
        'land_sea_mask', 'orography', 'lattitude', '2m_temperature',
        '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50',
        'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700',
        'geopotential_850', 'geopotential_925', 'geopotential_1000',
        'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500',
        'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850',
        'u_component_of_wind_925', 'u_component_of_wind_1000',
        'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500',
        'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850',
        'v_component_of_wind_925', 'v_component_of_wind_1000',
        'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600',
        'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000',
        'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500',
        'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850',
        'relative_humidity_925', 'relative_humidity_1000',
        'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500',
        'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850',
        'specific_humidity_925', 'specific_humidity_1000'
    ]
    
    out_vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_500",
        "temperature_850",
        "relative_humidity_500",
        "specific_humidity_850"
    ]
    
    pred_range = 1
    num_scales = 4
    num_global_operator_token = 16
    theta_cutoff = None
    
    # 加载数据
    print("加载数据...")
    logging.info("加载数据...")
    
    low_data = datamodule2(
        buffer_size=100,
        root_dir=args.low_res_root,
        variables=default_vars,
        out_variables=out_vars,
        batch_size=args.batch_size,
        predict_range=pred_range,
        num_workers=1
    )
    
    high_data = datamodule2(
        buffer_size=100,
        root_dir=args.high_res_root,
        variables=out_vars,
        out_variables=out_vars,
        batch_size=args.batch_size,
        predict_range=pred_range,
        num_workers=1
    )
    
    low_data.setup()
    high_data.setup()
    
    # 获取坐标
    low_lat, low_lon = low_data.get_lat_lon()
    lat, lon = high_data.get_lat_lon()
    
    # 获取高分辨率数据的维度
    # 从第一个批次获取实际的空间维度
    if args.split == "train":
        sample_loader = high_data.train_dataloader()
    elif args.split == "val":
        sample_loader = high_data.val_dataloader()
    else:  # test
        sample_loader = high_data.test_dataloader()
    
    # 获取一个样本来确定维度
    lat_high, lon_high = len(lat), len(lon)
    
    # 创建物理损失函数
    physics_loss_fn = None
    if lat_high > 0 and lon_high > 0:
        try:
            physics_loss_fn = create_physics_loss(
                nlat=lat_high,
                nlon=lon_high,
                grid="equiangular",
                divergence_weight=1.0,
                vorticity_weight=0.5,
                vapor_weight=0.5,
                use_divergence=True,
                use_vorticity=True,
                use_vapor=True,
                geopotential_weight=0.5,
                temperature_weight=0.5,
                use_geopotential=False,
                use_temperature=False,
            )
            physics_loss_fn = physics_loss_fn.to(device)
            print("物理损失函数已创建")
            logging.info("物理损失函数已创建")
        except Exception as e:
            print(f"警告: 创建物理损失函数失败: {e}")
            logging.warning(f"创建物理损失函数失败: {e}")
            physics_loss_fn = None
    
    # 获取归一化参数
    normalization = high_data.transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    denormalization = transforms.Normalize(mean_denorm, std_denorm)
    
    # 获取气候态
    if args.split == "test":
        test_clim = high_data.test_clim
    elif args.split == "val":
        test_clim = high_data.val_clim
    else:
        test_clim = None
    
    # 获取数据加载器
    if args.split == "train":
        low_dataloader = low_data.train_dataloader()
        high_dataloader = high_data.train_dataloader()
    elif args.split == "val":
        low_dataloader = low_data.val_dataloader()
        high_dataloader = high_data.val_dataloader()
    else:  # test
        low_dataloader = low_data.test_dataloader()
        high_dataloader = high_data.test_dataloader()
    
    # 生成坐标
    low_lat = torch.tensor(low_lat, dtype=torch.float32)
    low_lon = torch.tensor(low_lon, dtype=torch.float32)
    jj, kk = generate_coord(low_lat, low_lon)
    jjj, kkk = generate_coord(lat, lon)
    
    # 初始化模型和编码器
    print("初始化模型和编码器...")
    logging.info("初始化模型和编码器...")
    
    lat_weights = torch.cos(torch.deg2rad(low_lat))
    
    encoder = WeatherSphericalEncoder(
        in_channels=len(default_vars),
        lat_weights=lat_weights,
        num_scales=num_scales,
    )
    
    model = ClimaX(
        default_vars=default_vars,
        out_dim=7,
        low_gird=[jj, kk],
        high_gird=[jjj, kkk],
        num_global_operator_token=num_global_operator_token,
        theta_cutoff=theta_cutoff,
        dec_num_heads=4
    )
    
    encoder = encoder.to(device)
    model = model.to(device)
    
    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint}")
    logging.info(f"加载checkpoint: {args.checkpoint}")
    
    checkpoint_info = load_checkpoint_for_inference(
        args.checkpoint,
        model,
        encoder,
        device
    )
    
    print(f"Checkpoint信息:")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best Val Loss: {checkpoint_info['best_val_loss']}")
    print(f"  Best Val w_rmse: {checkpoint_info['best_val_w_rmse']}")
    print(f"  Best Val Epoch: {checkpoint_info['best_val_epoch']}")
    print(f"  Best Val w_rmse Epoch: {checkpoint_info['best_val_w_rmse_epoch']}")
    
    logging.info(f"Checkpoint信息: {checkpoint_info}")
    
    # 进行推理
    print(f"\n开始推理 ({args.split}集)...")
    logging.info(f"开始推理 ({args.split}集)...")
    
    metrics = inference(
        encoder=encoder,
        model=model,
        low_dataloader=low_dataloader,
        high_dataloader=high_dataloader,
        device=device,
        denormalization=denormalization,
        lat=lat,
        test_clim=test_clim,
        pred_range=pred_range,
        physics_loss_fn=physics_loss_fn,
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"推理结果 ({args.split}集):")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        print(f"  {key}: {value:.6f}")
        logging.info(f"{key}: {value:.6f}")
    
    # 保存结果到文件
    results_file = os.path.join(args.output_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write(f"推理结果 ({args.split}集)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint信息:\n")
        for k, v in checkpoint_info.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n指标:\n")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"\n结果已保存到: {results_file}")
    print(f"日志已保存到: {log_file}")
    logging.info(f"结果已保存到: {results_file}")


if __name__ == '__main__':
    main()

