import sys, pathlib
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
from torch.utils.data import Dataset
import hydra
from tqdm import tqdm
from functools import partial
import numpy as np

# 确保脚本能找到您的 Model 包
sys.path.append(os.getcwd())
from Model.collect import create_batch, replace_zeros_with_next_or_prev_nonzero, weighted_qsq, qsq_onball, mc_return_1traj
from Model.possession import *
from train import PlayerIDMapper

class SceneDataset(Dataset):
    """用于读取原始 .pkl 文件的 Dataset"""
    def __init__(self, root_dir):
        self.filepath_list  = glob.glob(os.path.join(root_dir, '*.pkl'), recursive=True)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        scene = None
        if os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'rb') as f:
                    scene = pickle.load(f)
            except pickle.UnpicklingError:
                print(f"Warning: Could not unpickle file {filepath}. Skipping.")
        return scene

def process_single_scene(scene, player_mapper, max_t=80, gamma=0.95, flag=0):
    """
    处理单个场景，返回处理后的数据字典
    这个函数基于 create_batch 的逻辑，但专门为单个场景设计
    """
    # 获取 hex indexer
    get_hex_index, _, _ = build_hex_indexer(r=1.5, rect_width=52, rect_height=52, padding_idx=598)
    
    T_possession = len(scene.timeframe)
    
    # 处理 agent ids
    ids_this = [player_mapper.get(agent.playerid) for agent in scene.agents]
    ids_this.append(500)  # ball id 0, rim id 500
    
    # 处理原始数据
    offense_tensors = []
    defense_tensors = []
    player_id = []

    for agent in scene.agents:
        hex_ids = [get_hex_index(xi, yi) for xi, yi in zip(agent.x, agent.y)]
        
        single_agent_array = np.array([
            [x, y, vx, vy, hex_id]
            for x, y, vx, vy, hex_id in zip(agent.x, agent.y, agent.vx, agent.vy, hex_ids)
        ])  # shape: [T, 5]

        single_agent_tensor = torch.tensor(single_agent_array.T, dtype=torch.float32).unsqueeze(0)  # shape: [1, 5, T]
        player_id.append(agent.playerid)

        if agent.teamid != scene.def_teamid:
            offense_tensors.append(single_agent_tensor)
        else:
            defense_tensors.append(single_agent_tensor)

    # 拼接成张量
    offense_tensor = torch.cat(offense_tensors, dim=0) if offense_tensors else None
    defense_tensor = torch.cat(defense_tensors, dim=0) if defense_tensors else None
    
    if offense_tensor is None or defense_tensor is None:
        return None  # 如果缺少进攻或防守球员，跳过这个场景
    
    scene_tensor = torch.cat([offense_tensor, defense_tensor], dim=0)  # shape: [11, 5, T]

    # 添加篮筐
    x_rim = 4.5
    y_rim = 25
    T = scene_tensor.shape[-1]
    hex_id_rim = get_hex_index(x_rim, y_rim)
    rim_array = torch.tensor(
        [[x_rim], [y_rim], [0.0], [0.0], [hex_id_rim]],
        dtype=torch.float32
    ).repeat(1, T)
    rim_tensor = rim_array.unsqueeze(0)

    # 最终拼接
    s = torch.cat([scene_tensor, rim_tensor], dim=0)  # [12, 5, T]
    player_id.append(500)

    # 降采样
    shot_frame = scene.outcome[2]
    indices = sorted(range(shot_frame-1, -1, -5))
    s = s[:,:,indices]
    
    # 重新排列维度
    s = s.permute(2, 0, 1)  # [T, 12, 5]

    T_orig, A, _ = s.shape
    if T_orig > 80 or T_orig < 30:
        return None  # 跳过不符合长度要求的场景
    
    # 处理动作
    a = scene.RL_data["actions"][:, 0:6] + 2
    mask = (a[:, 1:6] == 7)
    a[:, 1:6][mask] = 27
    a_ball = scene.RL_data['ball_action']
    if a.shape[0] == T_orig:
        a = a[:-1,:]
    a[:,0] = a_ball

    # 添加投篮行
    a = np.concatenate([a, np.array([[0, 27, 27, 27, 27, 27]])], axis=0)
    
    # 处理奖励
    r = scene.qSQ
    ball_status = scene.ball_status
    ball_status = np.concatenate([ball_status, ball_status[-1:]], axis=0)
    ball_status = replace_zeros_with_next_or_prev_nonzero(ball_status)
    ball_status = ball_status[indices]
    r = r[indices]
    
    # 处理边特征
    edge_index = scene.graph['ei2']
    edge_attr = scene.graph['ea2']
    edge_attr = np.concatenate([edge_attr, edge_attr[-1:, :, :]], axis=0)
    edge_attr = edge_attr[indices,:,:]
    
    # 奖励处理
    if flag == 0:  # on ball + off ball
        r_diff = weighted_qsq(r, ball_status)
    elif flag == 1:  # only on ball
        r_diff = qsq_onball(r, ball_status)
    elif flag == 2:  # sparse reward
        points = scene.outcome[-1]
        r_diff = np.zeros(T_orig, dtype=np.float32)
        r_diff[-1] = points
    
    mc = mc_return_1traj(r_diff.squeeze(), gamma=gamma)
    
    # Padding 到 MAX_T
    pad_state = np.full((max_t, A, 5), 1e9, dtype=np.float32)
    pad_state[:T_orig] = s.numpy()
    pad_state = np.transpose(pad_state, (1, 0, 2))  # (A, T, 5)

    pad_action = np.full((max_t, 6), 27, dtype=np.int64)
    pad_action[:T_orig] = a
    pad_action = pad_action.T  # (6, T)

    pad_mask = np.ones((A, max_t), dtype=bool)
    pad_mask[:, :T_orig] = False  # (A, T)

    pad_reward = np.full((max_t,), -1e9, dtype=np.float32)
    pad_reward[:T_orig] = r_diff.squeeze()

    pad_mc = np.full((max_t,), -1e9, dtype=np.float32)
    pad_mc[:T_orig] = mc.squeeze()

    pad_edge_attr = np.full((max_t, edge_attr.shape[1], 1), -1e9, dtype=np.float32)
    pad_edge_attr[:T_orig] = edge_attr

    # 转换为张量并返回
    return {
        "state_tokens": torch.tensor(pad_state, dtype=torch.float32),  # [A, T, 5]
        "action_tokens": torch.tensor(pad_action, dtype=torch.long),   # [6, T] -> 在 collate 时会变成 [B*6, T]
        "padding_mask": torch.tensor(pad_mask, dtype=torch.bool),      # [A, T]
        "agent_ids": torch.tensor(ids_this, dtype=torch.long),         # [A]
        "rewards": torch.tensor(pad_reward[None, :], dtype=torch.float32),  # [1, T]
        "done": torch.tensor([T_orig], dtype=torch.long),              # [1]
        "mc_return": torch.tensor(pad_mc[None, :], dtype=torch.float32),    # [1, T]
        "edge_index": torch.tensor(edge_index, dtype=torch.long),      # [2, 132]
        "edge_attr": torch.tensor(pad_edge_attr[None, :], dtype=torch.float32)  # [1, T, 132, 1]
    }

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    input_dir = '/root/autodl-fs/shotData_516'
    output_dir = '/root/autodl-fs/shotData_onlyball_preprocessed'
    os.makedirs(output_dir, exist_ok=True)

    dataset = SceneDataset(input_dir)
    player_mapper = PlayerIDMapper("/root/QT_NBA2/players.csv")

    print(f"Starting preprocessing of {len(dataset)} files...")
    processed_count = 0
    skipped_count = 0
    
    # 遍历所有原始场景文件
    for i in tqdm(range(len(dataset))):
        scene = dataset[i]
        if scene is None:
            skipped_count += 1
            continue
        
        try:
            # 使用新的单场景处理函数
            processed_data = process_single_scene(
                scene, 
                player_mapper, 
                max_t=80, 
                gamma=0.95, 
                flag=cfg.flag
            )
            
            if processed_data is None:
                skipped_count += 1
                continue

        except Exception as e:
            print(f"Skipping scene {i} (file: {dataset.filepath_list[i]}) due to error: {e}")
            skipped_count += 1
            continue

        # 将处理成功的结果使用 torch.save 保存
        output_filepath = os.path.join(output_dir, f'scene_{i}.pt')
        torch.save(processed_data, output_filepath)
        processed_count += 1

    player_mapper.save()
    print(f"\nFinished processing:")
    print(f"  - Successfully processed: {processed_count} scenes")
    print(f"  - Skipped: {skipped_count} scenes")
    print(f"  - Processed data saved to {output_dir}")

if __name__ == '__main__':
    main() 