import pandas as pd
import numpy as np
import torch
import random

import pandas as pd
import numpy as np
import torch
from .possession import *
from .utils import *
# -------------------------------------------------
# 1. 工具函数
# -------------------------------------------------

def replace_zeros_with_next_or_prev_nonzero(arr):
    arr = arr.copy()
    n = len(arr)
    # 第一次遍历，替换为后面第一个非零
    for i in range(n):
        if arr[i] == 0:
            for j in range(i + 1, n):
                if arr[j] != 0:
                    arr[i] = arr[j]
                    break
    # 第二次遍历，替换为前面最近非零（针对还为0的情况）
    for i in range(n):
        if arr[i] == 0:
            # 查找前面最近的非零
            for j in range(i-1, -1, -1):
                if arr[j] != 0:
                    arr[i] = arr[j]
                    break
    return np.array(arr)

def weighted_qsq(values: np.ndarray,ball_status) -> np.ndarray:
    """你之前写好的加权 QSQ + 差分"""
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("values 必须形如 [T,5]")
    ball_status = ball_status - 1 #1-5 -> 0-4
    T = values.shape[0]

    onball = values[np.arange(T), ball_status]

    mask = np.ones(values.shape, dtype=bool)
    mask[np.arange(T), ball_status] = False      # [T,5]，持球队员为 False
    offball = values[mask].reshape(T, 4)        # 先变一维，再 reshape


    #onball, offball = values[:, 0], values[:, 1:]
    row_sum = offball.sum(axis=1, keepdims=True)
    denom = np.where(row_sum < 1e-8, 1.0, row_sum)           # 防 0
    weights = offball / denom
    weights[row_sum.squeeze() < 1e-8] = 0.25                 # 均分
    means = (weights * offball).sum(axis=1)

    w_sum = 0.6 * onball + 0.4 * means
    diff = np.diff(w_sum)
    diff = np.append(diff, 0) 
    diff[-1] = onball[-1] - w_sum[-1]                        # 末帧修正
    return diff[:, None]                                     # [t,1]


def mc_return_1traj(rewards: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """单条轨迹 Monte‑Carlo Return，rewards shape [T] 或 [T,1]"""
    r = rewards.squeeze().astype(float).copy()               # 1‑D
    for t in range(len(r) - 2, -1, -1):
        r[t] += gamma * r[t + 1]
    return r[:, None]                                        # [T,1]


def qsq_onball(values: np.ndarray,ball_status) -> np.ndarray:
    """你之前写好的加权 QSQ + 差分"""
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("values 必须形如 [T,5]")

    ball_status = ball_status - 1 #1-5 -> 0-4
    T = values.shape[0]

    onball = values[np.arange(T), ball_status]
    #row_sum = offball.sum(axis=1, keepdims=True)
    #denom = np.where(row_sum < 1e-8, 1.0, row_sum)           # 防 0
    #weights = offball / denom
    #weights[row_sum.squeeze() < 1e-8] = 0.25                 # 均分
    #means = (weights * offball).sum(axis=1)

    #w_sum = 0.6 * onball + 0.4 * means
    diff = np.diff(onball)
    diff = np.append(diff, 0)
    diff[-1] = onball[-1] - 41.2                        # 末帧修正
    return diff[:, None]                                     # [T,1]

def add_xy_channels(grid_index_array, centers):
    """
    grid_index_array : ndarray (T,12)   取值 0~N_hex-1
    centers          : ndarray (N_hex,2)
    return           : (T,12,3)  (id, x, y)
    """
    T, A = grid_index_array.shape
    # 取中心坐标
    xy     = centers[grid_index_array.astype(int)]   # (T,12,2)
    state3d = np.empty((T, A, 3), dtype=np.float32)
    state3d[..., 0] = grid_index_array               # id
    state3d[..., 1:] = xy                            # x, y
    return state3d

# -------------------------------------------------
# 2. 主函数：把一个 scene list → batch 张量
# -------------------------------------------------
def create_batch(
    batch,                                  # list[Scene]
    player_mapper,
    max_t: int = 80,
    gamma: float = 0.95,
    bonus_if_made: float = 10.0,
    bonus_scale: float = 5.0,
    flag = 0,
):
    """
    返回 dict，键名符合 compute_loss 期望：
        state_tokens   : [B*A, T, 1] Float # 加入 x,y,Vx,Vy [x,y,Vx,Vy,hex_index]
        agent_ids      : [B*A]       Long
        padding_mask   : [B*A, T]    Bool
        action_tokens  : [B*A, T]    Long
        rewards        : [B,   T]    Float
        done           : [B]         Long  (每条轨迹有效长度)
        mc_return      : [B,   T]    Float
    """
    #定义 hex dict
    get_hex_index, _, _ = build_hex_indexer(r=1.5, rect_width=52, rect_height=52, padding_idx=598)


    # ---------- 各字段列表 ----------
    state_list, action_list, mask_list = [], [], []
    agent_id_list, done_len_list = [], []
    reward_list, mc_list = [], []
    edge_attr_list = []



    for scene in batch:

        T_possession = len(scene.timeframe) #totle moments frame 未降采样，且包含一些无效frame

        
        # ------------ 2.1 agent ids (11 个) ------------
        ids_this = [player_mapper.get(agent.playerid) for agent in scene.agents]

        ids_this.append(500) #ball id 0, rim id 500

        # ------------ 2.2 原始数据 ------------
        #s = scene.RL_data["states"]     # shape (T, 11) already down sampled
        offense_tensors = []
        defense_tensors = []
        player_id = []

        for agent in scene.agents:

            hex_ids = [get_hex_index(xi, yi) for xi, yi in zip(agent.x, agent.y)] #如果球员在右边半场，统一给index 600

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
        offense_tensor = torch.cat(offense_tensors, dim=0) if offense_tensors else None  # [6, 5, T]
        defense_tensor = torch.cat(defense_tensors, dim=0) if defense_tensors else None  # [5, 5, T]

        scene_tensor = torch.cat([offense_tensor, defense_tensor], dim=0)  # shape: [11, 5, T]

        #add rim detail
        x_rim = 4.5
        y_rim = 25
        T = scene_tensor.shape[-1]  # 时间步数
        hex_id_rim = get_hex_index(x_rim, y_rim)
        rim_array = torch.tensor(
            [[x_rim], [y_rim], [0.0], [0.0], [hex_id_rim]],  # shape: [5, 1]
            dtype=torch.float32
        ).repeat(1, T)  # shape: [5, T]
        rim_tensor = rim_array.unsqueeze(0)  # shape: [1, 5, T]

        # 最终拼接：shape = [12, 5, T]
        s = torch.cat([scene_tensor, rim_tensor], dim=0) #[12, 5, T]
       
        player_id.append(500) #add rim id

        #down sample 
        shot_frame = scene.outcome[2]
        indices = sorted(range(shot_frame-1, -1, -5))
        s = s[:,:,indices] 
         
        s = rearrange(s, 'a f t -> t a f')  #[T,12,5]

        T_orig, A,_ = s.shape                 # A=12. T_orig 是real input T
        if T_orig > 80:
            continue
        if T_orig < 30:
            continue
        
        a = scene.RL_data["actions"][:, 0:6] + 2 # 数据中球员移动的bin是6-24 ，改为8-26
        mask = (a[:, 1:6] == 7) 
        a[:, 1:6][mask] =  27 #无效数据 padding
        a_ball = scene.RL_data['ball_action'] # [0-7] 0:shot ,1-5 pass,6 dribble,7 on-air (need mask)
        if a.shape[0] == T_orig:
            a = a[:-1,:]
        a[:,0] = a_ball

        assert a.shape[0] == T_orig - 1

        

        # ------------ 2.3 动作add “投篮” 行 ------------
        a = np.concatenate([a, np.array([[0, 27, 27, 27, 27, 27]])], axis=0)  # (T,6)

        assert a.shape[0] == T_orig
        
        #r = scene.RL_data["rewards"]         # shape (T-1, 5)
        
        r = scene.qSQ # len(r) = T_possession
        ball_status = scene.ball_status #len(ball_status) shot_frame
        ball_status = np.concatenate([ball_status,  ball_status[-1:]], axis=0)
        ball_status = replace_zeros_with_next_or_prev_nonzero(ball_status) 

        ball_status = ball_status[indices]

        assert len(ball_status) == T_orig
        r = r[indices] 
        
        #把qsq也给state
        # s_3d = np.zeros((T_orig, A+1, 3), dtype=s.dtype) 
        # s_3d[:, :, 0] = s 
        # s_3d[:, 1:6, 2] = r   

        # crete edge feature
        edge_index = scene.graph['ei2'] #[2,132]
        edge_attr = scene.graph['ea2'] #[30,132,1] #没有down sampled
        edge_attr = np.concatenate([edge_attr,  edge_attr[-1: , :, :]], axis=0)
        edge_attr = edge_attr[indices,:,:]
        
        T_pad = max_t


        # ------------ 2.4 奖励处理 ------------
        #
        if flag == 0: # on ball + off ball
            r_diff = weighted_qsq(r,ball_status)                            # (T,1)
        if flag == 1: # only on ball
            print('only onball')
            r_diff = qsq_onball(r,ball_status)
        if flag == 2: #sparse reward 1, 0
            print('results')
            points = scene.outcome[-1]
            r_diff = np.zeros(T_orig, dtype=np.float32)  # 新建长度为 done 的全 0 np.array
            # 让最后一帧为 points，其余为 0:
            r_diff[-1] = points

        assert r_diff.shape[0] == a.shape[0]

        #if "MISS" not in str(scene.outcome[0]).upper():     # 命中 -> 加 bonus
            #r_diff[-1, 0] = (r_diff[-1, 0] + bonus_if_made) / bonus_scale
        
        mc = mc_return_1traj(r_diff.squeeze(), gamma=gamma)  # (T,1)

        # ------------ 2.6 padding 到 MAX_T ------------
        assert A == 12
        pad_state = np.full((T_pad, A, 5), 1e9, dtype=np.float32)
        pad_state[:T_orig] = s #[T,A,5]
        # 变换成 (A, T, 5) 以便后续 [B*A, T, 5]
        pad_state = np.transpose(pad_state, (1, 0, 2))      # (A, T, 5)

        pad_action = np.full((T_pad, 6), 27, dtype=np.int64) # 27 = 无效 ,模型中要nn.embedding
        pad_action[:T_orig] = a                             # (T,6)
        pad_action = pad_action.T                           # (6, T)

        pad_mask = np.ones((A, T_pad), dtype=bool)
        pad_mask[:, :T_orig] = False                        # (A, T)

        pad_reward = np.full((T_pad,), -1e9, dtype=np.float32)
        pad_reward[:T_orig] = r_diff.squeeze()              # (T,)

        pad_mc = np.full((T_pad,), -1e9, dtype=np.float32)
        pad_mc[:T_orig] = mc.squeeze()                      # (T,) 

        pad_edge_attr = np.full((T_pad, edge_attr.shape[1], 1), -1e9, dtype=np.float32)
        pad_edge_attr[:T_orig] = edge_attr

        # ------------ 2.7 收集 ------------
        state_list.append(pad_state)        # (A, T, 5)
        action_list.append(pad_action)      # (6, T)
        mask_list.append(pad_mask)          # (A, T)
        agent_id_list.append(ids_this)      # (A,)
        done_len_list.append(T_orig)        # 标量
        reward_list.append(pad_reward[None, :])  # (1, T)
        mc_list.append(pad_mc[None, :])          # (1, T)
        
        edge_attr_list.append(pad_edge_attr[None,:]) #(1,T,110,1)

# ---------- 3. batch 拼接 ----------
    # A = 11, 6 actions
    state_batch  = torch.tensor(np.concatenate(state_list , axis=0), dtype=torch.float32)  # (B*A, T, 5)
    action_batch = torch.tensor(np.concatenate(action_list, axis=0), dtype=torch.long)     # (B*6, T)
    mask_batch   = torch.tensor(np.concatenate(mask_list  , axis=0), dtype=torch.bool)     # (B*A, T)
    agent_batch  = torch.tensor(np.concatenate(agent_id_list), dtype=torch.long)           # (B*A,)
    done_batch   = torch.tensor(done_len_list, dtype=torch.long)                           # (B,)
    reward_batch = torch.tensor(np.concatenate(reward_list, axis=0), dtype=torch.float32)  # (B,T)
    mc_batch     = torch.tensor(np.concatenate(mc_list , axis=0), dtype=torch.float32)     # (B,T)
    edge_index_batch = torch.tensor(edge_index, dtype=torch.long)  #(2,110)
    edge_attr_batch = torch.tensor(np.concatenate(edge_attr_list , axis=0), dtype=torch.float32) #(B,T,110,1)

    # ---------- 4. 可选：保存更新后的 player 映射 ----------
    # player_df.to_csv(player_csv_path, index=False)

    return {
        "state_tokens":  state_batch,      # [B*A, T, 5]  padding = 1e9 ，grid index 598 = 后场
        "action_tokens": action_batch,     # [B*6, T]   padding = 27
        "padding_mask":  mask_batch,       # [B*A, T]
        "agent_ids":     agent_batch,      # [B*A]    
        "rewards":       reward_batch,     # [B,   T]  padding = -1e9
        "done":          done_batch,       # [B]    
        "mc_return":     mc_batch,         # [B,   T] padding = -1e9
        "edge_index":    edge_index_batch, # [2,132]
        "edge_attr":     edge_attr_batch   # [B,T,132,1] padding = -1e9
    }