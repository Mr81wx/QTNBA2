import math
from torch.distributions.laplace import Laplace
import torch
import torch.nn as nn
import torch.nn.functional as F
#from visualizer import get_local
import functools
import numpy as np
from einops import rearrange, pack, reduce, repeat
from torch_geometric.data import Data, Batch

from scipy.spatial import KDTree



class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Permute4Batchnorm(nn.Module):
    def __init__(self,order):
        super(Permute4Batchnorm, self).__init__()
        self.order = order
    
    def forward(self, x):
        assert len(self.order) == len(x.shape)
        return x.permute(self.order)

class ScaleLayer(nn.Module):

    def __init__(self, shape, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

    def forward(self, input):
        return input * self.scale

def init_xavier_glorot(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SelfAttLayer_Enc(nn.Module):
    def __init__(self, time_steps=80, feature_dim=128, head_num=4, k=4, across_time=True):
        super().__init__()
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num,add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)
         
    #@get_local('attention_map')
    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        #print(hidden_mask)
        A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A_,A__ = batch_mask.shape
        assert (A==A_ and A==A__)
        A___,T_ = padding_mask.shape
        assert (A==A___ and T==T_)

        x_ = self.layer_X_(x)                               # [A,T,D]

        if self.across_time:
            q_ = x_.permute(1,0,2)                          # [L,N,E] : [A,T,D]->[T,A,D]
            k,v = x_.permute(1,0,2), x_.permute(1,0,2)      # [S,N,E] : [A,T,D]->[T,A,D]

            key_padding_mask = padding_mask                 # [N,S] : [A,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,A,D]
            att_output, att_weight = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [A,T,D]
            
            # add attention map
            attention_map = att_weight
            
            att_output = att_output.permute(1,0,2)
        else:
            q_ = x_                                         # [L,N,E] = [A,T,D]
            k, v = x_, x_                                   # [S,N,E] = [A,T,D]

            key_padding_mask = padding_mask.permute(1,0)    # [N,S] = [T,A]
            attn_mask = batch_mask                          # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T,D]
            att_output, att_weight = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # add attention map
            attention_map = att_weight
            
        
        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(0).unsqueeze(-1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def batch_select_indices(q_values, actions):
    """
    安全选择 Q(s, a) 并处理 -1 padding。

    Args:
        q_values: [B*T, 6, num_bins] — Q-values for all actions per agent
        actions:  [B*A, T] — actions taken (agent flattened)
        num_bins: int — total number of discrete actions

    Returns:
        selected_q: [B*T, 6] — Q(s, a) for each agent
    """
    B_times_A, T = actions.shape
    B = B_times_A // 6  # 因为你是 6 个 agent
    num_bins = 25
    actions = rearrange(actions, '(b a) t -> (b t) a', a=6)  # [B*T, 6]

    # expand dims to gather: [B*T, 6, 1]
    action_indices = actions.unsqueeze(-1)  # [B*T, 6, 1]

    valid_mask = (action_indices >= 0) & (action_indices < num_bins)

    # 安全 gather：先把非法值临时设置成 0（任何合法 index 都行）
    safe_indices = action_indices.clone()
    safe_indices = torch.where(valid_mask, safe_indices, torch.zeros_like(safe_indices))
    
    # gather 正确的 Q 值
    q_selected = q_values.gather(dim=-1, index=safe_indices).squeeze(-1)  # [B*T, 6]

    # 把非法位置的 Q 值清零（或你可以设为 very low，比如 -1e6）
    q_selected = torch.where(valid_mask.squeeze(-1), q_selected, torch.zeros_like(q_selected))

    return q_selected

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def build_loss_mask_from_lengths(done_lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    done_lengths: [B]，每个样本的 episode 实际长度（例如 67 表示只有前67步有效）
    T: 所有样本统一的时间步数（padding 后的最大长度）
    
    返回: [B, T]，前 done 长度部分为 1，其余为 0
    """
    B = done_lengths.shape[0]
    device = done_lengths.device

    time_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    done_lengths = done_lengths.unsqueeze(1)                # [B, 1]

    mask = (time_idx < done_lengths).float()  # [B, T]
    return mask

def batch_gather_by_index(tensor: torch.Tensor, indices: torch.Tensor):
    """
    从 shape = [B, T] 的 tensor 中，提取每个 batch 第 indices[i] 行的值，返回 shape = [B]
    """
    B = tensor.size(0)
    return tensor[torch.arange(B, device=tensor.device), indices]

def make_edge_index_flag(device='cpu'):
    """
    0‑5、6‑10 组内边 attr=1，跨组边 attr=0（仍保留在图里）。
    """
    src, dst, w = [], [], []
    group1 = set(range(0, 6))
    group2 = set(range(6, 11))

    for i in range(11):
        for j in range(11):
            if i == j:          # 去掉自环
                continue
            src.append(i); dst.append(j)
            same = (i in group1 and j in group1) or (i in group2 and j in group2)
            w.append(1.0 if same else 0.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr  = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(-1)
    return edge_index, edge_attr

class ActionOneHotLinearEmbedding(nn.Module):
    def __init__(self, num_bins, embed_dim):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        self.proj = nn.Linear(num_bins, embed_dim)  # one-hot → dense

    def forward(self, action_tokens):
        """
        action_tokens: [B, A, T], int, ∈ [-1, num_bins - 1]
        -1 is padding token
        """
        # 1. 构建 mask: 1 if valid, 0 if -1 (padding)
        mask = (action_tokens != -1).float().unsqueeze(-1)  # [B, A, T, 1]

        # 2. clamp -1 to 0 (safe index), then one-hot encode
        action_clamped = action_tokens.clamp(min=0)  # [-1 → 0], others unchanged
        one_hot = F.one_hot(action_clamped, num_classes=self.num_bins).float()  # [B, A, T, num_bins]

        # 3. apply mask to zero out padding positions
        one_hot = one_hot * mask  # padding → zero vector

        # 4. project to embedding space
        action_embed = self.proj(one_hot)  # [B, A, T, embed_dim]

        return action_embed
    
def run_gat_framewise_with_batch(x_btad, gat_layer, edge_index, edge_attr):
    """
    使用 torch_geometric 的 Batch 模式，对每帧 (B, T) 的图进行 GATv2 处理。

    Args:
        x_btad: [B, T, A, D] - 每一帧的节点特征
        gat_layer: PyG 的 GATv2Conv 实例
        edge_index: [2,110]- 
        edge_attr: [B.T,110, 1] - 

    Returns:
        x_btad_out: [B, T, A, D] - GAT 处理后的节点特征
    """
    B, T, A, D = x_btad.shape
    x_list = x_btad.reshape(B * T, A, D)

    edge_attr = rearrange(edge_attr, 'b t e i -> (b t) e i')
    #print(edge_attr.shape)
    # 构建 PyG 的 Data list
    data_list = [
        Data(x=x_list[i], edge_index=edge_index, edge_attr=edge_attr[i])
        for i in range(B * T)
    ]
    batch = Batch.from_data_list(data_list)  # 自动 offset

    # 送入 GAT 层
    out = gat_layer(batch.x, batch.edge_index, batch.edge_attr)  # [B*T*A, D]

    # reshape 回 [B, T, A, D]
    x_btad_out = out.view(B, T, A, D)
    return x_btad_out

def get_q_actions_not_taken_full(q_preds, actions, valid_mask=None):
    """
    从 q_preds 中提取所有 Q 值，并将被选中的 bin 设置为 NaN。

    Args:
        q_preds:     [B*T, A, num_bins]
        actions:     [B*T, A] 或 [B*T, A, 1]
        valid_mask:  [B*T, A]，可选

    Returns:
        q_all:       [B*T, A, num_bins]，被选择的 bin 为 NaN，其余为原始 Q 值
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(-1)  # [B*T, A, 1]

    BTA, A, num_bins = q_preds.shape

    q_all = q_preds.clone()

    if valid_mask is not None:
        valid_mask = valid_mask & (actions.squeeze(-1) >= 0)
        actions = actions * valid_mask.unsqueeze(-1).float()  # 无效动作设为 0
    else:
        valid_mask = (actions.squeeze(-1) >= 0)

    # 构造索引
    bta_idx = torch.arange(BTA, device=q_preds.device).view(-1, 1).expand(-1, A)  # [B*T, A]
    agent_idx = torch.arange(A, device=q_preds.device).view(1, -1).expand(BTA, -1)  # [B*T, A]
    bin_idx = actions.squeeze(-1)  # [B*T, A]

    # 用三维索引将 taken 动作设为 NaN
    q_all = q_all.clone()
    q_all = q_all.scatter(dim=2, index=bin_idx.unsqueeze(-1), value=float('nan'))

    # 同样将无效 agent 位置全设为 NaN（可选）
    q_all = torch.where(valid_mask.unsqueeze(-1).expand_as(q_all), q_all, torch.full_like(q_all, float('nan')))

    return q_all  # [B*T, A, num_bins]



def adjust_edge_attr_single_sample_np(ball_states: np.ndarray,
                                      ball_targets: np.ndarray,
                                      edge_index: np.ndarray,
                                      base_edge_attr: np.ndarray) -> np.ndarray:
    """
    根据 ball_states 和 ball_targets，返回每帧的 edge_attr（只修改 ball→player 边）

    参数:
        ball_states:    [T]  int, 0=球在空中，1~5=控球者
        ball_targets:   [T]  int, 仅在 ball_states==0 时有效
        edge_index:     [2, E] int, 图边索引
        base_edge_attr: [E, 1] float, 静态边特征（如组内=1，跨组=0）

    返回:
        edge_attr_seq:  [T, E, 1] float，每帧图的边特征
    """
    T = ball_states.shape[0]
    E = edge_index.shape[1]

    edge_attr_seq = np.repeat(base_edge_attr[None, :, :], T, axis=0)  # [T, E, 1]

    # 建立 ball → player 的边索引映射（只处理 1~5）
    edge_map = {}
    for e in range(E):
        src, dst = edge_index[0, e], edge_index[1, e]
        if src == 0 and 1 <= dst <= 5:
            edge_map[dst] = e

    for t in range(T):
        state = ball_states[t]
        target = ball_targets[t]

        if state == 0 and 1 <= target <= 5:
            e_idx = edge_map.get(target)
            if e_idx is not None:
                edge_attr_seq[t, e_idx, 0] = 0.5  # 空中传球
        elif 1 <= state <= 5:
            e_idx = edge_map.get(state)
            if e_idx is not None:
                edge_attr_seq[t, e_idx, 0] = 1.0  # 控球连接

    return edge_attr_seq


def compute_margin_loss(q_pred_all_actions, margin=0.1, agent_start=1):
    """
    q_pred_all_actions: [B*T, 6, num_bins]
    margin: float, 最小 margin 值，推荐 0.05 ~ 0.2
    agent_start: 从哪个 agent index 开始应用（默认跳过 agent 0）

    return: margin loss scalar
    """
    BTA, num_agents, num_bins = q_pred_all_actions.shape

    # 1. 取出 bin=6 的 Q 值
    q_bin6 = q_pred_all_actions[:, agent_start:, 6]  # [B*T, num_agents-1]

    # 2. 取出除 bin=6 以外的所有 Q 值
    mask = torch.ones(num_bins, dtype=torch.bool, device=q_pred_all_actions.device)
    mask = mask.scatter(dim=0, index=torch.tensor([6], device=q_pred_all_actions.device), value=False)
    q_others = q_pred_all_actions[:, agent_start:, :]  # [B*T, agents, bins]
    q_others = q_others[..., mask]                     # [B*T, agents, 24]

    # 3. 找最大其他动作 Q
    q_max_others = q_others.max(dim=-1).values         # [B*T, agents]

    # 4. 计算 margin difference
    diff = q_bin6 + margin - q_max_others              # [B*T, agents]

    # 5. 只惩罚"bin6 比其他还高"的情况
    loss = F.relu(diff).mean()
    return loss

def cql_loss_per_head(q_preds, actions_cql, mask_cql, q_pred_device):
    # q_preds: [B*T, 6, 25]
    # actions_cql: [B*T, 6]
    mask_flat = mask_cql.reshape(-1)                          # [B*T]
    q_pred_flat = q_preds.reshape(-1, 6, 25)                  # [B*T, 6, 25]
    actions_flat = actions_cql.reshape(-1, 6)                 # [B*T, 6]

    q_pred_valid = q_pred_flat[mask_flat.bool()]              # [N, 6, 25]
    actions_valid = actions_flat[mask_flat.bool()]            # [N, 6]

    N = q_pred_valid.shape[0]
    logsumexp_q = torch.zeros(N, 6, device=q_pred_device)
    q_taken = torch.zeros(N, 6, device=q_pred_device)

    # agent 0
    logsumexp_q = logsumexp_q.scatter(dim=1, index=torch.tensor([0], device=q_pred_device).unsqueeze(0), src=torch.logsumexp(q_pred_valid[:, 0, :6], dim=-1).unsqueeze(1))
    q_taken = q_taken.scatter(dim=1, index=torch.tensor([0], device=q_pred_device).unsqueeze(0), src=q_pred_valid[:, 0].gather(-1, actions_valid[:, 0].unsqueeze(-1)).squeeze(-1).unsqueeze(1))
    agent0_is0 = (actions_valid[:, 0] == 0)
    agent0_weight = torch.where(agent0_is0, 0, 1.0)
    cql_agent0 = (logsumexp_q[:, 0] - q_taken[:, 0]) * agent0_weight

    cql_agents = [cql_agent0.unsqueeze(-1)]
    for agent in range(1, 6):
        logsumexp_q = logsumexp_q.scatter(dim=1, index=torch.tensor([agent], device=q_pred_device).unsqueeze(0), src=torch.logsumexp(q_pred_valid[:, agent, 6:], dim=-1).unsqueeze(1))
        bin_idx = actions_valid[:, agent] - 6
        q_taken = q_taken.scatter(dim=1, index=torch.tensor([agent], device=q_pred_device).unsqueeze(0), src=q_pred_valid[:, agent, 6:25].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1).unsqueeze(1))
        agent_action = actions_valid[:, agent]
        agent_weight = torch.where(agent_action > 6, 1e-4, 1.0)
        #agent_weight = torch.where(agent_action > 12, 1e-5, 1.0)
        cql_agent = (logsumexp_q[:, agent] - q_taken[:, agent]) * agent_weight
        cql_agents.append(cql_agent.unsqueeze(-1))
    
    return torch.cat(cql_agents, dim=-1).mean()



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



def get_relative_code(index1, index2):
    """根据 hex 编号差值，获取相对方向编码，支持奇偶列规则。"""
    delta = index2 - index1

    odd_position_rules = {
        0: 0, 24: 1, 25: 2, 1: 3, -24: 4, -1: 5, 23: 6, 48: 7, 49: 8,
        26: 9, 2: 10, -22: 11, -23: 12, -48: 13, -25: 14, -26: 15, -2: 16, 22: 17, 47: 18
    }

    even_position_rules = {
        0: 0, 24: 1, 1: 2, -23: 3, -24: 4, -25: 5, -1: 6, 48: 7, 25: 8,
        26: 9, 2: 10, -22: 11, -47: 12, -48: 13, -49: 14, -26: 15, -2: 16, 22: 17, 23: 18
    }

    if index1 % 2 == 1:
        code = odd_position_rules.get(delta, -1)
    else:
        code = even_position_rules.get(delta, -1)

    return code + 8 if code != -1 else -1  # -1 表示非法方向



def batch_ball_select_indices(q_values, actions):
    """
    安全选择 Q(s, a) 并处理 -1 padding。

    Args:
        q_values: [B*T, 1, 8] — Q-values for all actions per agent
        actions:  [B*A, T] — actions taken (agent flattened)
        num_bins: int — total number of discrete actions

    Returns:
        selected_q: [B*T, 6] — Q(s, a) for each agent
    """
    B_times_A, T = actions.shape
    B = B_times_A // 6  # 因为你是 6 个 agent
    
    actions = rearrange(actions, '(b a) t -> (b t) a', a=6)  # [B*T, 6]
    ball_action = actions[:,0:1] #[B*T, 1]

    num_bins = 8

    # expand dims to gather: [B*T, 1, 1]
    action_indices = ball_action.unsqueeze(-1)  # [B*T, 1, 1]

    valid_mask = (action_indices >= 0) & (action_indices < num_bins)

    # 安全 gather：先把非法值临时设置成 0（任何合法 index 都行）
    safe_indices = action_indices.clone()
    safe_indices = torch.where(valid_mask, safe_indices, torch.zeros_like(safe_indices))
    
    # gather 正确的 Q 值
    q_selected = q_values.gather(dim=-1, index=safe_indices).squeeze(-1)  # [B*T, 1]

    # 把非法位置的 Q 值清零（或你可以设为 very low，比如 -1e6）
    q_selected = torch.where(valid_mask.squeeze(-1), q_selected, torch.full_like(q_selected, -1e9))

    return q_selected # [B*T, 1]


def batch_player_select_indices(q_values, actions):
    """
    安全选择 Q(s, a) 并处理 -1 padding。

    Args:
        q_values: [B*T, 5, 19] — Q-values for all actions per agent
        actions:  [B*A, T] — actions taken (agent flattened)
        num_bins: int — total number of discrete actions

    Returns:
        selected_q: [B*T, 6] — Q(s, a) for each agent
    """
    B_times_A, T = actions.shape
    B = B_times_A // 6  # 因为你是 6 个 agent
    
    actions = rearrange(actions, '(b a) t -> (b t) a', a=6)  # [B*T, 6]
    player_actions = actions[:,1:] #[B*T, 5]

    num_bins = 19

    # expand dims to gather: [B*T, 6, 1]
    action_indices = player_actions.unsqueeze(-1)  # [B*T, 5, 1]

    # 判断动作索引是否在合法范围 8~26, True for 合法
    valid_mask = (action_indices >= 8) & (action_indices < 8 + num_bins)

    safe_indices = player_actions.clone()
    safe_indices = torch.where(valid_mask, safe_indices, torch.full_like(safe_indices, 8))  # 用8（合法最小动作）替代无效动作索引，避免报错

    # 映射到0~18范围索引
    safe_indices = safe_indices - 8

    # gather q值
    q_selected = q_values.gather(dim=-1, index=safe_indices).squeeze(-1)  # [B*T, 5]

    # padding位置赋极小值，方便loss忽略
    q_selected = torch.where(valid_mask.squeeze(-1), q_selected, torch.full_like(q_selected, -1e9))

    return q_selected # [B*T, 5] 



def cql_loss_hard_player(q_preds_player, player_action, min_reward, q_pred_device, extra_bins_weight=0.1):
    """
    安全选择 Q(s, a) 并处理 -1 padding。

    Args:
        q_preds_player: [B*T,5,19] 
        actions_player: [B*T,5] 实际的动作
        mask_cql: [B] 剔除padding 部分的loss计算
    Returns:
        loss
    """  
    BT,_ = player_action.shape
    num_bins = 19
    # 判断动作索引是否在合法范围 8~26, True for 合法
    valid_mask = (player_action >= 8) & (player_action < 8 + num_bins)

    safe_indices = player_action.clone()
    safe_indices = torch.where(valid_mask, safe_indices, torch.full_like(safe_indices, 8))  # 用8（合法最小动作）替代无效动作索引，避免报错

    # 映射到0~18范围索引
    safe_indices = safe_indices - 8

    # 构建动作 mask
    mask = torch.zeros(BT, 5, 19, dtype=torch.bool, device=q_pred_device)
    mask = mask.scatter(dim=-1, index=safe_indices.unsqueeze(-1), value=True)  # taken actions = True

    not_taken_mask = ~mask  # [B*T, 5, 19]

    # 惩罚所有未采样的动作 Q
    q_rest = q_preds_player  # [B*T, 5, 19]

    weights = torch.ones_like(q_rest, device=q_pred_device)
    weights = weights.scatter(dim=-1, index=torch.arange(1, 6, device=q_pred_device).unsqueeze(0).unsqueeze(0), value=extra_bins_weight)  # 对于 1-6 小惩罚
    weights = weights.scatter(dim=-1, index=torch.arange(7, 19, device=q_pred_device).unsqueeze(0).unsqueeze(0), value=extra_bins_weight * 0.5) #对于 7-24 更小惩罚

    penalty = ((q_rest - min_reward) ** 2) * weights * not_taken_mask.float()

    # 对 padding 部分进行mask , actions_player 中所有=27的都需要mask
    padding_mask = (player_action == 27)  # [B*T, 5]

    # broadcast 到 [B*T, 5, 19]（沿 bin 维度扩展）
    padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 19)

    # 置零 loss
    penalty = penalty * (~padding_mask).float() #  [B*T, 5, 19]

    return penalty

def cql_loss_hard_ball(q_preds_ball, actions_ball, min_reward, q_pred_device, extra_bins_weight=0.1):
    """
    安全选择 Q(s, a) 并处理 -1 padding。

    Args:
        q_preds_ball: [B*T,1,8] 
        actions_Ball: [B,T] 实际的动作
       
    Returns:
        loss
    """  
    
    B, T = actions_ball.shape
    actions_ball = actions_ball.reshape(B*T)  # [B*T]

    # --- 构建动作 mask ---
    num_bins = 8
    mask = torch.zeros(B*T, 1, num_bins, dtype=torch.bool, device=q_pred_device)
    
    valid_mask = (actions_ball >= 0) & (actions_ball < num_bins) #set padding 27 to 0
    safe_indices = actions_ball.clone()
    safe_indices = torch.where(valid_mask, safe_indices, torch.zeros_like(safe_indices))  # fallback to bin0
    mask = mask.scatter(dim=-1, index=safe_indices.unsqueeze(-1).unsqueeze(1), value=True)  # [B*T, 1, 8]

    not_taken_mask = ~mask  # [B*T, 1, 8]

    # --- 惩罚 Q ---
    weights = torch.ones_like(q_preds_ball, device=q_pred_device)
    weights = weights.scatter(dim=-1, index=torch.arange(1, 6, device=q_pred_device).unsqueeze(0).unsqueeze(0), value=extra_bins_weight)     # bin 1~6 小惩罚 传球
    weights = weights.scatter(dim=-1, index=torch.tensor([0], device=q_pred_device).unsqueeze(0).unsqueeze(0), value=0.0)                     # bin 0 不惩罚 投篮不惩罚

    penalty = ((q_preds_ball - min_reward) ** 2) * weights * not_taken_mask.float() #[B*T, 1, 8]

    # --- padding mask ---
    padding_mask = (actions_ball == 50).unsqueeze(-1).unsqueeze(-1)  # [B*T, 1, 1]
    padding_mask = padding_mask.expand(-1, -1, 8)                          # [B*T, 1, 8]
    penalty = penalty * (~padding_mask).float() #  [B*T, 1, 8]

    return penalty

def entropy_reg_ball(q_pred, actions_ball,reduced_coef=0.1): #对action = 6 的减小惩罚
    """
    Args:
        q_preds_ball: [B*T,1,8] 
        actions_ball: [B,T] 实际的动作
       
    Returns:
        loss
    """ 
    probs = F.softmax(q_pred, dim=-1)  # [B*T,1,8]
    log_probs = probs.clamp(min=1e-6).log()        # 防止 log(0)
    
    weights = torch.ones_like(probs)              # [B*T, 1, 8]
    weights = weights.scatter(dim=-1, index=torch.arange(6, 8, device=probs.device).unsqueeze(0).unsqueeze(0), value=reduced_coef)         # 让 bin6,7 的权重变小

    entropy_per_bin = - (probs * log_probs)       # [B*T, 1, 8]
    weighted_entropy = (entropy_per_bin * weights).sum(dim=-1)  # [B*T, 1]

    padding_mask = (actions_ball == 27).unsqueeze(-1)  # [B*T, 1]
    # 使用非原地操作避免梯度计算错误
    entropy_loss = weighted_entropy * (~padding_mask).float()

    return entropy_loss.mean()

def entropy_reg_player(q_pred, actions_player,reduced_coef=0.1): #对action = 6 的减小惩罚
    """
    Args:
        q_preds_player: [B*T,5,19] 
        actions_player: [B*T,5]
    Returns:
        loss
    """ 
    probs = F.softmax(q_pred, dim=-1)  # [B*T,5,19]
    log_probs = probs.clamp(min=1e-6).log()        # 防止 log(0)
    
    weights = torch.ones_like(probs)              # [B*T, 5, 19]
    weights = weights.scatter(dim=-1, index=torch.tensor([0], device=probs.device).unsqueeze(0).unsqueeze(0), value=reduced_coef)        # 让 bin 0 的权重变小

    entropy_per_bin = - (probs * log_probs)       # [B*T, 5, 19]
    weighted_entropy = (entropy_per_bin * weights).sum(dim=-1)  # [B*T, 5]

    padding_mask = (actions_player == 27)  # [B*T, 5]
    # 使用非原地操作避免梯度计算错误
    entropy_loss = weighted_entropy * (~padding_mask).float()

    return  entropy_loss.mean()






