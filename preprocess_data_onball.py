import sys, pathlib
import os
import glob
import warnings
import torch
import pickle
from torch.utils.data import Dataset
import hydra
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
from einops import rearrange


# 确保脚本能找到您的 Model 包
sys.path.append(os.getcwd())
from Model.possession import *


import numpy as np

# ===== 节点编号约定 =====
BALL = 0
OFF_START, OFF_END = 1, 5      # 进攻 1..5
DEF_START, DEF_END = 6, 10     # 防守 1..5
RIM = 11                       # 篮筐
A_DEFAULT = 12                 # ball + 5 off + 5 def + rim

# ============= 静态有向边：不含自环 =============
def build_edge_index_np(A: int = A_DEFAULT) -> np.ndarray:
    """
    返回 edge_index: [2, E] (int64)，为 A 个节点的有向全连接图，去掉自环。
    """
    src = np.repeat(np.arange(A, dtype=np.int64), A)
    dst = np.tile(np.arange(A, dtype=np.int64), A)
    mask = src != dst
    return np.stack([src[mask], dst[mask]], axis=0)   # [2, E]

def edge_lookup(A: int, edge_index: np.ndarray) -> np.ndarray:
    """
    (u, v) -> 边列索引 的查表矩阵，shape [A, A]，不存在记为 -1。
    """
    lut = -np.ones((A, A), dtype=np.int64)
    s, d = edge_index
    for k in range(s.shape[0]):
        lut[s[k], d[k]] = k
    return lut

# ============= 动态 edge_attr（单通道 0/1 掩码） =============
def _next_receiver_from_bs(bs: np.ndarray, t: int) -> int | None:
    """
    从 t 之后寻找第一个非 0 的 ball_status（1..5）作为接球者，找不到则 None。
    """
    for u in range(t + 1, len(bs)):
        if OFF_START <= bs[u] <= OFF_END:
            return int(bs[u])
    return None

def build_edge_attr_with_ball_np(
    ball_status: np.ndarray,
    edge_index: np.ndarray,
    A: int = A_DEFAULT,
    airborne_to_rim_if_terminal: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    生成 edge_attr: [T, E, 1] (float32)
    规则：
      - 若 bs[t] = a ∈ {1..5}（有进攻球员持球）：
          a -> ball 置 1   （表明谁在持球）
          ball -> offense(1..5) 置 1 （下一步可传）
          ball -> rim 置 1           （可投篮）
        其余边 0；不使用自环（运球由动作头区分）
      - 若 bs[t] = 0（空中）：
          仅 ball <-> receiver 置 1（receiver 为之后第一个非 0 的持球者）
          若找不到且 airborne_to_rim_if_terminal=True，则 ball -> rim 置 1（终止兜底）
    """
    bs = np.asarray(ball_status, dtype=np.int64)
    T = len(bs)
    assert edge_index.shape[0] == 2, "edge_index 必须是 [2, E]"

    lut = edge_lookup(A, edge_index)
    E = edge_index.shape[1]
    ea = np.zeros((T, E, 1), dtype=dtype)
    off_nodes = np.arange(OFF_START, OFF_END + 1, dtype=np.int64)

    def set_edge(t: int, u: int, v: int):
        k = lut[u, v]
        if k != -1:
            ea[t, k, 0] = 1.0

    for t in range(T):
        b = bs[t]
        if OFF_START <= b <= OFF_END:
            # 谁在持球：a -> ball
            set_edge(t, b, BALL)
            # 下一步可传/可投：ball -> off(1..5), ball -> rim
            for o in off_nodes:
                set_edge(t, BALL, o)
            set_edge(t, BALL, RIM)
        else:
            # 空中：ball ↔ receiver
            rcv = _next_receiver_from_bs(bs, t)
            if rcv is not None:
                set_edge(t, BALL, rcv)
                set_edge(t, rcv, BALL)
            elif airborne_to_rim_if_terminal:
                set_edge(t, BALL, RIM)

    return ea  # [T, E, 1]

# =============（可选）节点类型 one-hot，便于同构 GAT 识别角色 =============
def build_node_type_onehot_np(T: int, A: int = A_DEFAULT, dtype=np.float32) -> np.ndarray:
    """
    返回 node_attr: [T, A, 4]，类型 one-hot：ball/off/def/rim
    """
    types = np.zeros((A, 4), dtype=dtype)
    types[BALL, 0] = 1.0
    types[OFF_START:OFF_END+1, 1] = 1.0
    types[DEF_START:DEF_END+1, 2] = 1.0
    types[RIM, 3] = 1.0
    return np.broadcast_to(types, (T, A, 4)).copy()

# ============= 示例 =============

def mc_return_1traj(rewards: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """单条轨迹 Monte‑Carlo Return，rewards shape [T] 或 [T,1]"""
    r = rewards.squeeze().astype(float).copy()               # 1‑D
    for t in range(len(r) - 2, -1, -1):
        r[t] += gamma * r[t + 1]
    return r[:, None]                                        # [T,1]

def adjust_ball_actions(ball_actions):
    
    # 找到最后一个非0动作的位置
    nonzero_indices = np.where(ball_actions != 0)[0]

    if len(nonzero_indices) == 0:
        # 全是0，只保留最后一个
        return np.array([0])

    last_nonzero_idx = nonzero_indices[-1]
    last_action = ball_actions[last_nonzero_idx]

    # 构建新数组
    trimmed = ball_actions.copy()

    # 把最后一个非0之后的所有0（除了最后一个）替换成最后一个非0动作
    if last_nonzero_idx + 1 < len(ball_actions):
        trimmed[last_nonzero_idx + 1:-1] = last_action

    return trimmed

from scipy.spatial import KDTree

def build_hex_indexer(r=1.5, rect_width=52, rect_height=52, padding_idx=600):
    dx = 3 / 2 * r
    dy = np.sqrt(3) * r

    def get_hexagonal_grid(cols, rows, r):
        grid = []
        for row in range(rows):
            for col in range(cols):
                x = col * dx
                y = row * dy
                if col % 2 == 1:
                    y += dy / 2
                grid.append((x, y))
        return grid

    # 计算需要的 hex 网格数量
    cols = int(np.ceil(rect_width / dx)) + 1
    rows = int(np.ceil(rect_height / dy)) + 1
    hex_centers = get_hexagonal_grid(cols, rows, r)

    # 为每个中心分配 index
    hex_index_map = {center: idx for idx, center in enumerate(hex_centers)}

    # 构建 KDTree 一次
    hex_tree = KDTree(hex_centers)

    # 定义查找函数
    def get_hex_index(x, y):
        if x > 54 :
            return padding_idx
        _, idx = hex_tree.query([x, y])
        return idx  # 已与 hex_centers 顺序一致

    return get_hex_index, hex_index_map, hex_centers


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

def process_single_scene(scene, player_mapper, max_t=80, gamma=0.95):
    """
    处理单个场景，返回处理后的数据字典
    这个函数基于 create_batch 的逻辑，但专门为单个场景设计
    """
    # 获取 hex indexer
    #get_hex_index, _, _ = build_hex_indexer(r=1.5, rect_width=52, rect_height=52, padding_idx=598)
    
    T_possession = len(scene.timeframe)
    
    # 处理 agent ids
    ids_this = [player_mapper.get(agent.playerid) for agent in scene.agents]
    ids_this.append(500)  # ball id 0, rim id 500
    
    # 处理原始数据
    player_id = []

    for agent in scene.agents:
        player_id.append(agent.playerid)
    
    x = torch.from_numpy(scene.RL_data['states']).float() # [T,11]
    scene_tensor = x.permute(1, 0).unsqueeze(1).contiguous() # [11, 1, T]


    A, F, T = scene_tensor.shape                 # A=11, F=1

    # 计算篮筐的 hex 索引
    x_rim, y_rim = 4.5, 25.0
    hex_id_rim = 218

    # 用 scene_tensor 的 dtype/device 构造 rim 节点，形状 [1, 1, T]
    rim_tensor = torch.full(
        (1, F, T),                               # F 要和 scene_tensor 的特征数一致（这里 =1）
        fill_value=hex_id_rim,
        dtype=scene_tensor.dtype,
        device=scene_tensor.device,
    )

    # 拼接到第 0 维（agent 维度）：[11,1,T] -> [12,1,T]
    s = torch.cat([scene_tensor, rim_tensor], dim=0)   # [A, F, T]
    player_id.append(500)  # rim id 500
    
    # 重新排列维度
    s = s.permute(2, 0, 1)  # [T, A, F]

    T_orig, _, _ = s.shape
    if T_orig > 80 or T_orig < 10:
        return None  # 跳过不符合长度要求的场景
    
    # 处理动作
    a = scene.RL_data["actions"][:, 0:6]
    
    # mask = (a[:, 1:6] == 7)
    # a[:, 1:6][mask] = 27
    # a_ball = scene.RL_data['ball_action']
    # if a.shape[0] == T_orig:
    #     a = a[:-1,:]
    # a[:,0] = a_ball

    # 添加投篮行
    a_full = np.concatenate([a, np.array([[0, 6, 6, 6, 6, 6]])], axis=0)
    a_full[:,0] = adjust_ball_actions(a_full[:, 0])
    
    # 处理奖励
    ball_actions = a_full[:-1, 0] #[71]
    assert len(ball_actions) == T_orig - 1

    #rewards = scene.RL_data['rewards']   # [72, 5]
    rewards = scene.qsq[::-5][::-1] 
    if len(rewards) != T_orig:
        return None

    # 取“球”的动作列，并在前面补1个元素与 rewards 对齐到 72
    idx_raw  = ball_actions.astype(np.int64)         # [71]
    idx_raw_ = np.concatenate([idx_raw[:1], idx_raw]) # [72]

    # 1..5 -> 0..4 的列索引；其余视为无效
    cols  = idx_raw_ - 1
    valid = (idx_raw_ >= 1) & (idx_raw_ <= 5)

    # 无效位置先临时用 0 列占位，取完再置 NaN
    cols_safe = np.where(valid, cols, 0)

    # 直接按列索引提取，得到 
    picked = np.take_along_axis(rewards, cols_safe[:, None], axis=1).astype(np.float32)  #
    picked[~valid] = np.nan  # 非传球/非法值 → NaN

    # 可选：安全检查
    assert picked.shape == (rewards.shape[0], 1)

    q_last = picked[-1]

    r_diff = picked[1:] - picked[0:-1]

    made = scene.RL_data['made'] 

    r_shot = 0.5 * float(made) + (q_last - 40) / 10 # 投篮奖励

    r_full = np.concatenate([r_diff, [r_shot]])  # [T, 1]

    assert r_full.shape == (T, 1)

    mc_full = mc_return_1traj(r_full.squeeze(), gamma=gamma)

    #获得ball_status
    shot_frame = scene.outcome[2]
    indices = sorted(range(shot_frame, -1, -5))

    ball_status = scene.ball_status[indices]

    if len(ball_status) != T_orig:
        print(len(ball_status))
        print(T_orig)
    
    # 处理边特征
    # 静态边（无自环）
    edge_index = build_edge_index_np(A_DEFAULT)          # [2, E]
    # 动态边特征（单通道 0/1 掩码）
    edge_attr  = build_edge_attr_with_ball_np(
        ball_status=ball_status,
        edge_index=edge_index,
        A=A_DEFAULT,
        airborne_to_rim_if_terminal=True,
        dtype=np.float32
    )                                                       # [T, E, 1]

    
    # Padding 到 MAX_T
    PAD_HEX =598  # hex indexer 的 padding idx
    F = 1         # 特征数（hex index）
    PAD_ACTION = 30  # 动作空间外的值
    PAD_REWARD = -1e9
    MAX_T = 80  # 你传入的 max_t
    A = A_DEFAULT
    E = edge_index.shape[1]

    state_pad = torch.full((MAX_T, A, F), fill_value=float(PAD_HEX),
                       dtype=s.dtype, device=s.device)
    state_pad[:T] = s
    state_tokens = state_pad.permute(1, 0, 2).contiguous()         # [A, MAX_T, F]

    action_pad = np.full((MAX_T, 6), fill_value=PAD_ACTION, dtype=np.int64)
    action_pad[:T] = a_full
    action_tokens = torch.from_numpy(action_pad).long().transpose(0, 1).contiguous()  

    # reward: [T,1] -> pad -> [MAX_T,1]
    reward_pad = np.full((MAX_T, 1), fill_value=PAD_REWARD, dtype=np.float32)
    reward_pad[:T] = r_full
    rewards_tensor = torch.from_numpy(reward_pad).float()          # [MAX_T,1]

    # mc return: [T] -> pad -> [MAX_T]
    mc_pad = np.full((MAX_T,1), fill_value=PAD_REWARD, dtype=np.float32)
    mc_pad[:T] = mc_full
    mc_return_tensor = torch.from_numpy(mc_pad).float().unsqueeze(0)  # [1, MAX_T]

    # edge_attr: [T,E,1] -> pad -> [MAX_T,E,1]
    edge_attr_pad = np.zeros((MAX_T, E, 1), dtype=np.float32)
    edge_attr_pad[:T] = edge_attr
    edge_attr_tensor = torch.from_numpy(edge_attr_pad).float().unsqueeze(0)  # [1, MAX_T, E, 1]

    # edge_index: numpy -> torch
    edge_index_tensor = torch.from_numpy(edge_index).long()     # [2, E]

    # padding mask: [A, MAX_T]（True=pad；False=有效）
    padding_mask = torch.ones((A, MAX_T), dtype=torch.bool)
    padding_mask[:, :T] = False

    # agent_ids（长度要和 A 一致；这里做个稳妥拼接）
    ids_this = [player_mapper.get(agent.playerid) for agent in scene.agents]  # 通常=10个球员
    if len(ids_this) == 10:
        ids_this = ids_this + [0, 500]   # + ball id(0) + rim id(500)
    elif len(ids_this) == 11:
        ids_this = ids_this + [500]      # 已含球，再加 rim
    agent_ids = torch.tensor(ids_this, dtype=torch.long)

    # done（真实有效长度）
    done = torch.tensor([T], dtype=torch.long)

    # qsq
    qsq = rewards  #[T,5]
    qsq = rearrange(qsq, 't a -> a t') #[5,T]
    qsq_pad = np.full((5, MAX_T), fill_value=PAD_REWARD, dtype=np.float32)
    qsq_pad[:, :T] = qsq
    qsq_tensor = torch.from_numpy(qsq_pad).float()  # [5, MAX_T]

    processed_data = {
    "id": scene.gameid + '_' + scene.id,
    "state_tokens": state_tokens,                    # [A, MAX_T, F]
    "action_tokens": action_tokens,                  # [6, MAX_T]
    "padding_mask": padding_mask,                    # [A, MAX_T]
    "agent_ids": agent_ids,                          # [A]
    "rewards": rewards_tensor.transpose(0, 1),       # [1, MAX_T]
    "done": done,                                    # [1]
    "mc_return": mc_return_tensor,                   # [1, MAX_T]
    "edge_index": edge_index_tensor,                 # [2, E]
    "edge_attr": edge_attr_tensor,                   # [1, MAX_T, E, 1]
    "qsq": qsq_tensor                                 # [5, MAX_T]
}

    return processed_data

    # # 转换为张量并返回
    # return {
    #     "state_tokens": torch.tensor(pad_state, dtype=torch.long),  # [A, T, 1]
    #     "action_tokens": torch.tensor(pad_action, dtype=torch.long),   # [6, T] -> 在 collate 时会变成 [B*6, T]
    #     "padding_mask": torch.tensor(pad_mask, dtype=torch.bool),      # [A, T]
    #     "agent_ids": torch.tensor(ids_this, dtype=torch.long),         # [A]
    #     "rewards": torch.tensor(pad_reward[None, :], dtype=torch.float32),  # [1, T]
    #     "done": torch.tensor([T_orig], dtype=torch.long),              # [1]
    #     "mc_return": torch.tensor(pad_mc[None, :], dtype=torch.float32),    # [1, T]
    #     "edge_index": torch.tensor(edge_index, dtype=torch.long),      # [2, 132]
    #     "edge_attr": torch.tensor(pad_edge_attr[None, :], dtype=torch.float32)  # [1, T, 132, 1]
    # }

def main():
    input_dir = '/root/autodl-fs/shotData_516'
    output_dir = '/root/autodl-fs/shotData_onball_preprocessed2'
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
            RL_data = scene.RL_data
        except:
            continue
        
        try:
            # 使用新的单场景处理函数
            processed_data = process_single_scene(
                scene, 
                player_mapper, 
                max_t=80, 
                gamma=0.95
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