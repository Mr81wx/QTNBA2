import sys, pathlib
from torch.utils.data import Dataset
import os
os.chdir('/mnt/nvme_share/srt02/EPV_Transformer/QT_NBA2')
import glob
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
import tensorboard
from torch.utils.data import Dataset, DataLoader,random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from Model.QT_PL import QT
from Model.collect import *

from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from Model.possession import*
from functools import partial


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_float32_matmul_precision('medium')

# train.py

import os
import pandas as pd

class PlayerIDMapper:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame(columns=["playerid", "UnifiedPlayerID"])
        self.id_map = dict(zip(self.df["playerid"], self.df["UnifiedPlayerID"]))
        self.next_id = int(self.df["UnifiedPlayerID"].max() + 1) if not self.df.empty else 0

    def get(self, playerid: int) -> int:
        if playerid in self.id_map:
            return self.id_map[playerid]
        else:
            new_id = self.next_id
            self.id_map[playerid] = new_id
            self.df.loc[len(self.df)] = {"playerid": playerid, "UnifiedPlayerID": new_id}
            self.next_id += 1
            return new_id

    def save(self):
        self.df.to_csv(self.csv_path, index=False)

class ProcessedSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list  = glob.glob(os.path.join(root_dir, '*.pt'), recursive=True)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        return torch.load(filepath)

def simple_collate_fn(batch):
    # This is a basic collate function. It assumes that each element of the batch 
    # is a dictionary of tensors from our pre-processing script.
    # You might need to adjust this based on the exact structure of your processed data.
    elem = batch[0]
    return {key: torch.cat([d[key] for d in batch], dim=0) for key in elem}
    
def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = list(sorted(glob.glob(os.path.join(checkpoint_dir, '*.ckpt'), recursive=True)))
    return checkpoint_paths[-1] if checkpoint_paths else None


import pandas as pd



@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    torch.cuda.empty_cache()
    pl.seed_everything(1235)

    dataset = ProcessedSceneDataset('/mnt/nvme_share/srt02/EPV_Transformer/QT_NBA/shotData_516_processed') #路径设置

    # PlayerIDMapper is no longer needed during training, as IDs are processed offline.
    # player_mapper = PlayerIDMapper("/mnt/nvme_share/srt02/EPV_Transformer/QT_NBA/players.csv")


    #dataset = SceneDataset('/content/shotData')
    train_size = int(len(dataset) * 0.8)  # 80%作为训练集
    val_size = len(dataset) - train_size  # 剩余部分作为验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # The collate function is now much simpler
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batchsize, shuffle=True,num_workers=8,collate_fn=simple_collate_fn, drop_last=True,pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batchsize, shuffle=False,num_workers=8,collate_fn=simple_collate_fn, drop_last=True, pin_memory=True)

    #set checkpoint
    checkpoint_path = 'checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        monitor='val/q_mean',
        dirpath=checkpoint_path,
        filename='model-{epoch:02d}-{val_q_mean:.4f}',
        save_top_k=3,
        mode='max'
    )
    #train process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_ids)

    model = QT(cfg)  
    # in_feat_dim,time_steps,feature_dim,head_num,k,F,halfwidth,temperature,batchsize,lr,tasks
    tb_logger = TensorBoardLogger("logs/", name= cfg.check_point_name)
    #tb_callback = TensorBoardCallback()
    
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         logger=tb_logger,
                         callbacks=[checkpoint_callback],
                         enable_checkpointing=True,                   
                         accelerator="gpu"
                         )
    trainer.fit(model, train_dataloader, val_dataloader,
               )

    # player_mapper.save() is also not needed here anymore.

        
if __name__ == '__main__':
    sys.exit(main())