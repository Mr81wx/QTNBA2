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


class ProcessedSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list = sorted(glob.glob(f"{root_dir}/*.pt"))
        

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        return torch.load(filepath)

# 2. 简单的collate_fn
def simple_collate_fn(batch):
    """
    batch: list of scenes, 每个 scene 是一个 dict，包含 processed_data
    """
    # 取出 processed_data
    pd_list = [scene["processed_data"] for scene in batch]

    out = {}
    keys = pd_list[0].keys()

    for key in keys:

        vals = [pd[key] for pd in pd_list]

        # 1. edge_index — 所有 sample 相同，不做 batch
        if key == "edge_index":
            out[key] = vals[0]
            continue

        # 2. 如果不是 tensor（可能是字符串等），直接返回列表
        if not isinstance(vals[0], torch.Tensor):
            out[key] = vals
            continue

        # --- 根据 key 决定 cat / stack ---
        if key in ["state_tokens", "action_tokens", "padding_mask", "agent_ids","qsq", "rewards", "done", "edge_attr"]:
            # 这些是 [A, T, ...] 或 [A] 维度
            out[key] = torch.cat(vals, dim=0)

        else:
            raise ValueError(f"Unknown processed_data field: {key}")

    return out



@hydra.main(config_path="conf", config_name="config0103.yaml")


def main(cfg):
    pl.seed_everything(1235)

    # 3. 直接加载处理好的数据
    dataset = ProcessedSceneDataset('/root/autodl-fs/clean_pt')
    
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

    check_point_name = cfg.check_point_name if 'check_point_name' in cfg else 'QT_model'

    checkpoint_path = '/root/autodl-tmp/QT_checkpoints/' + check_point_name
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val/acc_ball',
    #     dirpath=checkpoint_path,
    #     filename='model-{epoch:02d}-{val_q_mean:.4f}',
    #     save_top_k=3,
    #     mode='max'
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='model-{epoch:02d}-{val_acc_ball:.4f}',  # 或 val_q_mean_ball
        save_top_k=-1,     # ✅ 保存所有 epoch
        every_n_epochs=15,  # ✅ 每个 epoch 保存一次
        monitor='val/acc_ball',  # 可以保留或删除（此时不影响保存逻辑）
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
    trainer.save_checkpoint(os.path.join(checkpoint_path, "model-last.ckpt"))


if __name__ == '__main__':
    main() 