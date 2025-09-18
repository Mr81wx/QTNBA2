import glob
import torch
import time
from torch.utils.data import Dataset, DataLoader
import os

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list = glob.glob(f"{root_dir}/*.pt")
        print(f"Found {len(self.filepath_list)} data files")

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.filepath_list[idx], map_location='cpu')
            return data
        except Exception as e:
            print(f"Error loading file {self.filepath_list[idx]}: {e}")
            return None

def test_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    elem = batch[0]
    out = {}
    
    for key in elem:
        if key == 'edge_index':
            out[key] = elem[key]
        else:
            try:
                out[key] = torch.cat([d[key] for d in batch], dim=0)
            except Exception as e:
                print(f"Error in collate for key {key}: {e}")
                out[key] = elem[key]
    
    return out

def main():
    print("Testing data loading...")
    
    # 检查数据目录
    data_dir = '/mnt/nvme_share/srt02/EPV_Transformer/QT_NBA/shotData_516_processed'
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        return
    
    # 创建数据集
    dataset = TestDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # 测试单个样本加载
    print("\nTesting single sample loading...")
    start_time = time.time()
    sample = dataset[0]
    load_time = time.time() - start_time
    print(f"Single sample load time: {load_time:.3f}s")
    
    if sample is not None:
        print(f"Sample keys: {sample.keys()}")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, {v.dtype}")
    else:
        print("Sample is None!")
        return
    
    # 测试DataLoader
    print("\nTesting DataLoader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=test_collate_fn, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        timeout=30
    )
    
    # 测试几个batch
    print("Testing batch loading...")
    start_time = time.time()
    batch_count = 0
    for i, batch in enumerate(dataloader):
        if batch is None:
            print(f"Batch {i} is None!")
            continue
            
        batch_count += 1
        print(f"Batch {i}: {len(batch)} keys")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, {v.dtype}")
        
        if batch_count >= 3:  # 只测试前3个batch
            break
    
    total_time = time.time() - start_time
    print(f"Loaded {batch_count} batches in {total_time:.3f}s")
    print(f"Average time per batch: {total_time/batch_count:.3f}s")

if __name__ == '__main__':
    main() 