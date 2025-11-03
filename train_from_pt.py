import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from Model.QT_PL import QT
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import time

# 移除异常检测以提高性能
# torch.autograd.set_detect_anomaly(True)

# 1. 新的数据集类
class ProcessedSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list = glob.glob(f"{root_dir}/*.pt")

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        return torch.load(self.filepath_list[idx])

# 2. 简单的collate_fn
def simple_collate_fn(batch):
    elem = batch[0]
    out = {}
    keys = list(batch[0].keys())[1:]
    for key in keys:
        if key == 'edge_index':
            # edge_index 对所有样本都是一样的，只取第一个即可
            out[key] = elem[key]
        else:
            # 其他字段正常拼接
            out[key] = torch.cat([d[key] for d in batch], dim=0)
    return out

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    pl.seed_everything(1235)

    # 3. 直接加载处理好的数据
    dataset = ProcessedSceneDataset('/root/autodl-fs/shotData_onball_preprocessed2')
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试数据加载
    print("Testing data loading...")
    try:
        sample_data = dataset[0]
        print(f"Sample data keys: {sample_data.keys()}")
        for k, v in sample_data.items():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                print(f"  {k}: {v.shape}, {v.dtype}")
            else:
                print(f"  {k}: type={type(v)} -> skipped (non-tensor)")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 测试 collate_fn
    print("Testing collate function...")
    try:
        sample_data = dataset[0]
        print(f"Sample data keys: {sample_data.keys()}")
        for k, v in sample_data.items():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                print(f"  {k}: {v.shape}, {v.dtype}")
            else:
                print(f"  {k}: type={type(v)} -> skipped (non-tensor)")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return

    # 优化数据加载配置
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.batchsize, 
        shuffle=True, 
        num_workers=8,  # 增加worker数量
        collate_fn=simple_collate_fn, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2  # 预取因子
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.batchsize, 
        shuffle=False, 
        num_workers=8,  # 增加worker数量
        collate_fn=simple_collate_fn, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2  # 预取因子
    )

    checkpoint_path = '/root/autodl-tmp/QT_checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor='val/l1_q_vs_mc',
        dirpath=checkpoint_path,
        filename='model-{epoch:02d}-{val_q_mean:.4f}',
        save_top_k=3,
        mode='max'
    )

    tb_log_dir = '/root/tf-logs'
    os.makedirs(tb_log_dir, exist_ok=True)
    model = QT(cfg)
    tb_logger = TensorBoardLogger(tb_log_dir, name=cfg.check_point_name)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        accelerator="gpu",
        devices=1,  # 明确指定设备数量
        precision="16-mixed",  # 启用混合精度训练
        gradient_clip_val=1.0,  # 梯度裁剪
        accumulate_grad_batches=1,  # 梯度累积
        sync_batchnorm=False,  # 禁用同步BN
        deterministic=False,  # 禁用确定性模式以提高性能
        benchmark=True,  # 启用cudnn benchmark
    )
    
    print("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main() 