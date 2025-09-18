import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from Model.QT_PL import QT
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import os
import gc

# 禁用异常检测以提高性能
# torch.autograd.set_detect_anomaly(True)

# 设置环境变量优化性能
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 禁用同步执行
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # 启用cuDNN v8

# 1. 优化的数据集类
class ProcessedSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list = glob.glob(f"{root_dir}/*.pt")
        print(f"Found {len(self.filepath_list)} data files")

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.filepath_list[idx], map_location='cpu')
            # 确保数据类型正确
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].float()
            return data
        except Exception as e:
            print(f"Error loading file {self.filepath_list[idx]}: {e}")
            # 返回一个空的数据结构
            return None

# 2. 优化的collate_fn
def optimized_collate_fn(batch):
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    elem = batch[0]
    out = {}
    
    for key in elem:
        if key == 'edge_index':
            # edge_index 对所有样本都是一样的，只取第一个即可
            out[key] = elem[key]
        else:
            # 其他字段正常拼接
            try:
                out[key] = torch.cat([d[key] for d in batch], dim=0)
            except Exception as e:
                print(f"Error in collate for key {key}: {e}")
                # 如果拼接失败，取第一个样本
                out[key] = elem[key]
    
    return out

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    pl.seed_everything(1235)

    # 设置设备
    device = torch.device(f"cuda:{cfg.device_ids}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 检查GPU内存
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

    # 3. 直接加载处理好的数据
    dataset = ProcessedSceneDataset('/mnt/nvme_share/srt02/EPV_Transformer/QT_NBA/shotData_516_processed')
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试数据加载
    print("Testing data loading...")
    try:
        sample_data = dataset[0]
        if sample_data is not None:
            print(f"Sample data keys: {sample_data.keys()}")
            for k, v in sample_data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, {v.dtype}")
        else:
            print("Warning: Sample data is None")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 测试 collate_fn
    print("Testing collate function...")
    try:
        test_batch = [dataset[i] for i in range(min(4, len(dataset)))]
        test_batch = [item for item in test_batch if item is not None]
        if test_batch:
            collated = optimized_collate_fn(test_batch)
            if collated is not None:
                print(f"Collated batch keys: {collated.keys()}")
                for k, v in collated.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}, {v.dtype}")
            else:
                print("Warning: Collated data is None")
        else:
            print("Warning: No valid data in test batch")
    except Exception as e:
        print(f"Error in collate function: {e}")
        return

    # 优化数据加载配置
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.batchsize, 
        shuffle=True, 
        num_workers=8,  # 根据CPU核心数调整
        collate_fn=optimized_collate_fn, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2,  # 预取因子
        timeout=60  # 增加超时时间
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.batchsize, 
        shuffle=False, 
        num_workers=8,  # 根据CPU核心数调整
        collate_fn=optimized_collate_fn, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2,  # 预取因子
        timeout=60  # 增加超时时间
    )

    checkpoint_path = 'checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        monitor='val/q_mean',
        dirpath=checkpoint_path,
        filename='model-{epoch:02d}-{val_q_mean:.4f}',
        save_top_k=3,
        mode='max',
        save_last=True  # 保存最后一个checkpoint
    )

    model = QT(cfg)
    tb_logger = TensorBoardLogger("logs/", name=cfg.check_point_name)

    # 优化训练器配置
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        accelerator="gpu",
        devices=[cfg.device_ids],  # 明确指定设备
        precision="16-mixed",  # 启用混合精度训练
        gradient_clip_val=1.0,  # 梯度裁剪
        accumulate_grad_batches=1,  # 梯度累积
        sync_batchnorm=False,  # 禁用同步BN
        deterministic=False,  # 禁用确定性模式以提高性能
        benchmark=True,  # 启用cudnn benchmark
        enable_progress_bar=True,
        log_every_n_steps=10,  # 减少日志频率
        val_check_interval=0.5,  # 每半个epoch验证一次
        num_sanity_val_steps=2,  # 减少sanity check步数
        reload_dataloaders_every_n_epochs=0,  # 不重新加载数据
    )
    
    print("Starting training...")
    print(f"Batch size: {cfg.batchsize}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except Exception as e:
        print(f"Training failed: {e}")
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    main() 