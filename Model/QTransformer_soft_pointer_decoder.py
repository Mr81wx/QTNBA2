from einops import rearrange, pack, reduce, repeat
from dataclasses import dataclass
import torch
import copy
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from random import random
#from utils import init_weights, default, batch_select_indices, zero_
#from gpt_like import Transformer

from .utils import *
from torch_geometric.nn import GATv2Conv

def upgrade_edge_attr(edge_attr_old, ball_status, edge_index):
    """
    输入：
        edge_attr_old: [T, E, 1] —— 你之前版本构造好的，只包含 pass 信息
        ball_status:   [T]       —— 表示每一帧谁持球（0=空中，1..5=进攻持球）
        edge_index:    [2, E]    —— 静态有向边

    输出：
        edge_attr_new: [T, E, 3] —— 新 3 维 edge_attr
    """

    ea_old = edge_attr_old  # [T,E,1]
    T, E, _ = ea_old.shape
    s, d = edge_index       # [E], [E]

    BALL = 0
    OFF  = np.arange(1, 6)     # 1..5
    DEF  = np.arange(6, 11)    # 6..10
    RIM  = 11

    # 新的 edge_attr 容器
    ea_new = np.zeros((T, E, 3), dtype=np.float32)

    # ========================================================
    # channel 0 —— team relation
    # ========================================================
    team_rel = np.zeros(E, dtype=np.float32)

    for k in range(E):
        u, v = s[k], d[k]

        # ball 与任何节点 = 0
        if u == BALL or v == BALL:
            team_rel[k] = 0
            continue

        # rim 视为 OFFENSE
        u_is_off = (u in OFF) or (u == RIM)
        v_is_off = (v in OFF) or (v == RIM)

        # 同队=0, 不同队=1
        team_rel[k] = float(u_is_off != v_is_off)

    # broadcast 复制
    ea_new[:, :, 0] = team_rel


    # ========================================================
    # channel 1 —— ball handler: u -> Ball
    # ========================================================
    for t in range(T):
        b = ball_status[t]
        if 1 <= b <= 5:
            mask = (s == b) & (d == BALL)
            ea_new[t, mask, 1] = 1.0


    # ========================================================
    # channel 2 —— pass options（简化）
    # Ball -> Offense(1..5) 和 Ball -> Rim
    # ========================================================
    pass_mask = (s == BALL) & ((d >= 1) & (d <= 5) | (d == RIM))
    ea_new[:, pass_mask, 2] = 1.0

    return ea_new


def _init_linear_kaiming(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class SelfAttLayerCausalTime(nn.Module):
    def __init__(self, time_steps=80, feature_dim=128, head_num=4, dropout=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num

        self.layer_norm_in = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=head_num, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU()
        )
        self.layer_norm_out = nn.LayerNorm(feature_dim)

    def forward(self, x, padding_mask=None):
        """
        x: [B*A, T, D] — input tensor
        padding_mask: [B*A, T] — True where padding (optional)
        """
        _, T, D = x.shape
        assert T == self.time_steps and D == self.feature_dim
        x_norm = self.layer_norm_in(x)  # [B*A, T, D]
        #print(" 41 has NaN?", torch.isnan(x_norm).any())
        # Create causal mask: [T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Multihead attention
        attn_output, _ = self.attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask  # [B*A, T], True for padding
        )  # Output: [A, T, D]
        #print(" 52 has NaN?", torch.isnan(attn_output).any())
        x_res = attn_output + x
        x_ffn = self.ffn(x_res)
        x_out = self.layer_norm_out(x_ffn)
        #print(" 56 has NaN?", torch.isnan(x_out).any())

        return x_out

#10.14 Encoder onlyGAT
class Encoder(nn.Module):
    def __init__(
        self, device,
        in_feat_dim=1, time_steps=80,
        feature_dim=128, head_num=6,
        n_player_ids=600                # 0号留给球
    ):
        super().__init__()
        self.device = device
        self.T, self.D = time_steps, feature_dim
        self.A, self.H = 12, head_num
        assert feature_dim % head_num == 0

        #
        self.hex_embedding = nn.Embedding(600,12,padding_idx=598) #0-503 有效值 504-599 padding 598
        #self.hex_project = nn.Linear(4,12)

        # ---- Embedding ----
        self.player_embedding = nn.Embedding(n_player_ids, 12)

        #--- index Embedding ----
        self.index_embedding = nn.Embedding(12,12)

        # ---- FFN (layer A) ----
        # self.layer_A = nn.Sequential(
        #     nn.Linear(12, 32),
        #     nn.BatchNorm1d(32), nn.LeakyReLU(),
        #     nn.Linear(32, 128),
        #     nn.BatchNorm1d(128), nn.LeakyReLU(),
        #     nn.Linear(128, feature_dim),
        #     nn.BatchNorm1d(feature_dim), nn.LeakyReLU(),
        # )

        self.layer_A = nn.Sequential(nn.Linear(36,64), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(64), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(64,256), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(256), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(256,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU() )
        self.layer_A.apply(init_xavier_glorot)

        # ---- PosEnc + Self‑Att ----
        #self.posenc  = PositionalEncoding(feature_dim, 0.1, time_steps)
        #self.selfatt1 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        #self.selfatt2 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        #self.selfatt3 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)

        # ---- GATv2Conv（帧内）----
        self.gat1 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=3, concat=False)
        self.gat2 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=3, concat=False)
        self.gat3 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=3, concat=False)

        self.norm1 = nn.LayerNorm(self.D)
        self.norm2 = nn.LayerNorm(self.D)
        self.norm3 = nn.LayerNorm(self.D)


    # ---------------- forward ----------------
    def forward(self, state_feat, padding_mask, agent_ids,edge_index,edge_attr):
        """
        state_feat : (B*A, T, 1) [hex_index]
        padding_mask: (B*A, T) # if add spatial-attention
        agent_ids  : (B*A,)
        edge_index: [B,T]
        edge_attr: [B,T,132,1]
        """

        N, T, _ = state_feat.shape
        B = N // self.A
        assert T == self.T

        #embedding states to 12
        #x,y,vx,vy
        #state_con = state_feat[...,:4]
        #state_xy = self.input_project(state_con)  # shape: [..., 12]
        #hex
        hex_id = state_feat[..., -1]
        hex_id  = hex_id.squeeze(-1)
        hex_id = torch.where(hex_id == 1e9, torch.full_like(hex_id, 599), hex_id) #实际有效值为0-503,padding 值 599,右边半场无效为598
        hex_emb  = self.hex_embedding(hex_id.long()) #[B*A,T,12] 

        #qsq = state_feat[..., 1:]
        #qsq = self.qsq_project(qsq) #[B*A,T,12]

        # ---- Player Embedding ----
        player_emb = self.player_embedding(agent_ids.clamp(min=0))  # [B*A, 12]
        player_emb = player_emb.unsqueeze(1).expand(-1, T, -1)             # [N, T, 12]

        # ---- Index Embedding ----
        index_ids = torch.arange(self.A, device=self.device).view(1, self.A, 1).expand(B, self.A, T)  # [B, A, T]
        index_emb = self.index_embedding(index_ids).reshape(B * self.A, T, -1)    

        x = torch.cat([ hex_emb, player_emb, index_emb], dim=-1) #[B*A,T,12*3]

        # ---- Layer A ----
        #x = rearrange(x, 'n t d -> (n t) d')    # [N*T, D]
        x = self.layer_A(x)                     # e.g. Linear, MLP, LayerNorm...
        #x = rearrange(x, '(n t) d -> n t d', n=N, t=T)  # [N, T, D]
        
        # ---- PosEnc & Self‑Att + 三次 GAT ----
        #x = self.posenc(x) 不需要位置编码

        
        for  gat, norm in zip([self.gat1, self.gat2, self.gat3], [self.norm1, self.norm2, self.norm3]):
            #x = self_att(x, padding_mask)                    # [N, T, D]

            x_btad = x.view(B, self.A, T, self.D).permute(0, 2, 1, 3)  # [B, T, A, D]

            x_btad = run_gat_framewise_with_batch(
                x_btad, gat,
                edge_index,
                edge_attr
            )

            x_ = x_btad.permute(0, 2, 1, 3).reshape(N, T, self.D)  # [N, T, D]

            x = norm(x + x_)   # Residual connection + LayerNorm

           

        return x #[N, T, D]
 

#decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.drop2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop3 = nn.Dropout(dropout)  # residual dropout

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, return_attn_weights=False):
        """
        x: [T, B, D] — decoder input tokens (ball + players)
        memory: [S, B, D] — prefix memory (state + prev ids + actions)
        tgt_mask: [T, T] — causal mask for decoder self-attn
        memory_mask: [T, S] — prefix mask for cross-attn
        return_attn_weights: bool — whether to return attention weights
        """

        # 1. Cross-attention to prefix memory
        x_norm = self.ln2(x)
        cross_attn_output, cross_attn_weights = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        #x = x + cross_attn_output

        x = x + self.drop1(cross_attn_output)



        # 2. Causal self-attention
        x_norm = self.ln1(x)
        self_attn_output, self_attn_weights = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        #x = x + self_attn_output
        x = x + self.drop2(self_attn_output)


        # 3. Feedforward
        x_norm = self.ln3(x)
        ff_out = self.ff(x_norm)
        x = x + self.drop3(ff_out)

        if return_attn_weights:
            return x, self_attn_weights, cross_attn_weights
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=128, num_heads=8, dropout=0.1,max_seq_len=64):
        super().__init__()
        #self.pos_emb_decoder = nn.Embedding(max_seq_len, embed_dim)
        #self.pos_emb_memory  = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, return_all_attn=False):
        """
        x: [T, B, D] — decoder input token sequence (ball + players)
        memory: [S, B, D] — encoder output (prefix sequence)
        tgt_mask: [T, T] — causal mask for self-attn
        memory_mask: [T, S] — prefix mask for cross-attn
        return_all_attn: bool — whether to return all attention weights from each layer
        """
        T, B, D = x.shape 
        S = memory.size(0)

        # Add positional encoding
        #pos_ids_x = torch.arange(T, device=x.device).unsqueeze(1)
        #pos_ids_mem = torch.arange(S, device=memory.device).unsqueeze(1)
        #x = x + self.pos_emb_decoder(pos_ids_x).expand(-1, B, -1)
        #memory = memory + self.pos_emb_memory(pos_ids_mem).expand(-1, B, -1)
        
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(x.device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(x.device)


        all_self_weights = []
        all_cross_weights = []      
        
        for layer in self.layers:
            if return_all_attn:
                x, self_weights, cross_weights = layer(
                    x, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    return_attn_weights=True
                )
                all_self_weights.append(self_weights)
                all_cross_weights.append(cross_weights)
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)

        if return_all_attn:
            return (x, all_self_weights, all_cross_weights)
        return x 


class ball_pointer_head(nn.Module):
    """
    Pointer-style Q-head for ball actions.
    输入:
        ball_h:    [B, T, D]         每个时间步的 ball token 表示 (通常是 encoder/decoder 输出的 x[..., 0, :])
        player_h:  [B, T, Np, D]     每个时间步 Np 个进攻球员的 token 表示 (x[..., 1:6, :])
        rim_h:     [B, T, D]         每个时间步 rim token 表示 (x[..., -1, :])
    输出:
        q_ball:    [B, T, Np+1]      对 receiver = {rim(0), P1(1), ..., P_Np(Np)} 的 Q 值

    """
    def __init__(self, d_model: int, num_players: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_players = num_players

        # Q, K 映射
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(num_players + 1, num_players + 1)
        self.scale = d_model ** 0.5
    
    def forward(
        self,
        ball_h: torch.Tensor,            # [B, T, D]
        player_h: torch.Tensor,          # [B, T, Np, D]
        rim_h: torch.Tensor               # [B, T, D]
    ) -> torch.Tensor:
        B, T, D = ball_h.shape
        _, _, Np, _ = player_h.shape
        assert Np == self.num_players, "num_players mismatch"

        rim_h = rim_h.unsqueeze(2)  # [B, T, 1, D]

        # 2) concat rim + players => memory keys
        # mem: [B, T, 1+Np, D]
        mem = torch.cat([rim_h, player_h], dim=2) #[rim =0, p1=1, p2=2, p3=3, p4=4, p5=5] #[B,T,6,D]

        # 3) flatten for batch matmul
        # ball_flat: [B*T, D]
        ball_flat = rearrange(ball_h, 'b t d -> (b t) d')
        # mem_flat:  [B*T, 6, D]
        mem_flat  = rearrange(mem,     'b t n d -> (b t) n d')

        q = self.q_proj(ball_flat).unsqueeze(1)   # [B*T, 1, D]
        k = self.k_proj(mem_flat)                 # [B*T, 6, D]

        logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        logits = logits.squeeze(1)  # [B*T, 6]

        logits_proj = self.out_proj(logits)
        # 6) reshape back
        a_ball = rearrange(logits_proj, "(b t) n -> b t n", b=B, t=T)

        return a_ball #[B, T, 6]



def build_prefix_mask(T, prefix_lens, total_mem_len, device):
    """
    T: decoder token 数（如 6)
    prefix_lens: list of int, 每个 decoder token 可看到的 memory 长度
    total_mem_len: memory 总长度（如 11)
    return: mask of shape [T, S], bool 类型,True 表示 mask(不可见)
    """
    mask = torch.ones(T, total_mem_len, dtype=torch.bool, device=device)
    for i in range(T):
        mask = mask.scatter(dim=1, index=torch.arange(prefix_lens[i], device=device).unsqueeze(0), value=False)
    return mask





class QTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.num_actions = cfg.num_actions
        self.num_bins = cfg.num_bins
        self.embed_dim = cfg.embed_dim 
        self.max_tokens = cfg.max_tokens
        #self.num_agents = cfg.num_agents
        self.num_bins_player = cfg.num_bins_player
        self.num_bins_ball = cfg.num_bins_ball



        self.state_embedding = Encoder(self.device, in_feat_dim=cfg.in_feat_dim, time_steps=cfg.time_steps, feature_dim=self.embed_dim, head_num=cfg.encoder_n_head)
        # self.identify_embedding = nn.Sequential(
        #                             nn.Linear(12, 64),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(64, self.embed_dim)
        #                             )
        #self.action_bin_embeddings = nn.Embedding(cfg.num_bins+1, cfg.embed_dim,padding_idx=30) #有一个padding
        self.ball_action_bin_embeddings = nn.Embedding(cfg.num_bins+1, cfg.embed_dim,padding_idx=30) #有一个padding
        self.player_action_bin_embeddings = nn.Embedding(cfg.num_bins+1, cfg.embed_dim,padding_idx=30) #有一个padding


        self.posenc  = PositionalEncoding(self.embed_dim, 0.1, 12)

        self.fuse_layer = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.decoder = Decoder(
            num_layers=cfg.decoder_layers,
            embed_dim=self.embed_dim,
            num_heads=cfg.n_head,
            dropout=cfg.dropout,
            max_seq_len=self.max_tokens
        )

        self.decoder_qsq = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1)
        )

        self.Value_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, 1)
        )

        self.ball_pointer_head_1 = ball_pointer_head(d_model=self.embed_dim, num_players=5)
        self.ball_pointer_head_2 = ball_pointer_head(d_model=self.embed_dim, num_players=5)
        
        self.A_pl_head_1 = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, self.num_bins_player)
        )
        self.A_pl_head_2 = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, self.num_bins_player)
        )

        self.ball_cls_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 2)   # num_classes = 2 (dribble / not dribble)
        )
        
        # self.Advantage = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim // 2, 1)
        #     )



    def forward(self, state_tokens, agent_ids, padding_mask, action_tokens,edge_index,edge_attr):
        '''state_tokens [B*A,T,1]
           agent_ids [B*A] 
           padding_mask [B*A,T]
           action_tokens [B*A,T]
           edge_index: [2,132]
           edge_attr: [B,T,132,1]
           '''
        
        
        # Encode Step
        state_feat = self.state_embedding(state_tokens, padding_mask, agent_ids,edge_index,edge_attr) #[B*A,T,D]
        #print("state_feat NaN:", torch.isnan(state_feat).any())
        state_feat = rearrange(state_feat, '(b a) t d -> b a t d', a=12)
        state_embed = state_feat.mean(dim=1) # [B, T, D]

        B, T, D = state_embed.shape

        player_embed_ = state_feat[:,:6,:,:] #[B,6,T,D]
        
        #Decoder Step
       
        #player embedding [B,D]
        #player_embed = rearrange(player_embedding, '(b a) t d -> b a t d', a=12)
        #player_embed_ = player_embed[:,:6,:,:] #[B,6,T,D]
        #player_embed_ = self.identify_embedding(player_embed_) #[B,6,T,D]

        #player embedding [B,T,D] 直接用node feature 
        #player_embed_ = state_feat[:,:6,:,:] # [B,6,T,D] 5.25修改
        
        #action action_tokens [B,6,T,1]
        
        action_tokens_ = rearrange(action_tokens, '(b n) t -> b n t',n=6) # [B,6,T]
        ball_action_tokens = action_tokens_[:,0:1,:] #[B,1,T]
        player_action_tokens = action_tokens_[:,1:,:] #[B,5,T]

        ball_action_embed = self.ball_action_bin_embeddings(ball_action_tokens)  # [B, 1, T, D]
        player_action_embed = self.player_action_bin_embeddings(player_action_tokens)  # [B, 5, T, D]
       # print(action_tokens_.dtype)
        #print("action_tokens:", action_tokens_.shape)
        action_embed = torch.cat([ball_action_embed, player_action_embed], dim=1)
        #action_embed = self.action_bin_embeddings(action_tokens_) # [B,6,T,D]
        #action_embed_ = rearrange(action_embed, '(b n t) d -> b n t d', b = Batch_size,n=6,t=T)# B,6,T,D

        S = rearrange(state_embed, 'b t d -> (b t) d')  # B*T,D
        
        
        
        P = rearrange(player_embed_, 'b a t d -> (b t) a d') # B*T,6,D #ball  o1 o2 o3 o4 o5 player embedding
        assert P.shape == (B*T, 6, D)
        A = rearrange(action_embed , 'b a t d -> (b t) a d') # B*T,6,D

        #A = torch.cat([P,A],dim = -1) # B*T,6,D*2
        #A = self.fuse_layer(A) # B*T,6,D

        seq = torch.zeros(B*T, 12, D, device=P.device)

        for i in range(6):
            seq[:, i*2, :] = P[:, i, :]    # 偶数位置放P
            seq[:, i*2+1, :] = A[:, i, :]  # 奇数位置放A

        # # 拼接
        # SPA = torch.cat([S, A], dim=1) 
        # assert SPA.shape == (B*T, 7, D)

        
        #decoder_input = self.posenc(P) # [B*T,6,D]
        decoder_input = self.posenc(seq) # [B*T,12,D]
        decoder_input  = rearrange(decoder_input, 'b a d -> a b d')     # [12, B*T, D]
        
        #prefix_memory  = rearrange(SPA, 'b s d -> s b d')   # [7, B*T, D]
        #prefix_memory = prefix_memory[ :-1 , :, : ]  # 不需要最后一个球员的动作 # [6, B*T, D]
        prefix_memory = rearrange(state_feat, 'b a t d -> a (b t) d')   # [12, B*T, D]
        assert prefix_memory.shape == (12, B*T, D)
        # prefix_mask = build_prefix_mask(
        #     T = self.num_actions,
        #     prefix_lens=[2 * i + 1 for i in range(self.num_actions)],
        #     total_mem_len=1 + 2 * self.num_actions,
        #     device=self.device
        # )
        
        #prefix_mask = torch.triu(torch.ones(12, 12, dtype=torch.bool, device=self.device), diagonal=1)
        prefix_mask = torch.zeros(12, 12, dtype=torch.bool, device=self.device)

        causal_mask = torch.triu(torch.ones(self.num_actions *2 , self.num_actions *2, device=self.device), diagonal=1).bool()
        # Decode
        decoder_out = self.decoder(decoder_input, prefix_memory, tgt_mask=causal_mask, memory_mask=prefix_mask)
        #decoder_out [12, B*T, D]
        ball_h = decoder_out[0,...]  # [B*T, D]
        ball_h = rearrange(ball_h, '(b t) d -> b t d', b=B, t=T)  # [B, T, D]
        player_h = prefix_memory[1:6,...]  # [5,B*T, D]
        player_h = rearrange(player_h, 'n (b t) d -> b t n d', b=B, t=T)  # [B, T, 5, D]    
        rim_h = prefix_memory[-1,...]  # [B*T, D]
        rim_h = rearrange(rim_h, '(b t) d -> b t d', b=B, t=T)  # [B, T, D]
        
        Value = self.Value_head(state_embed)  #[B,T,1]
        Value = rearrange(Value, 'b t d -> (b t) d')  # [B*T,1]

        a_ball_1 = self.ball_pointer_head_1(
            ball_h,          # [B,T, D]
            player_h,    # [B,T, 5, D]
            rim_h           # [B,T, D]   
        )  # [B,T, 6]

        a_ball_2 = self.ball_pointer_head_2(
            ball_h,          # [B,T, D]
            player_h ,    # [B,T, 5, D]
            rim_h           # [B, T, D]   
        )  # [B,T, 6]

        a_ball_1 = rearrange(a_ball_1,'b t d -> (b t) 1 d') # [B*T,1,6]
        a_ball_2 = rearrange(a_ball_2,'b t d -> (b t) 1 d') # [B*T,1,6]
        #print('a_ball_1',a_ball_1.shape)
        decoder_out_ = decoder_out[0::2, ...]  # [6, B*T, D]
        decoder_out_player = decoder_out_[1:6, ...]  # [5, B*T, D]
        a_pl_1 = self.A_pl_head_1(decoder_out_player)  # [5, B*T, 19]
        a_pl_2 = self.A_pl_head_2(decoder_out_player)  # [5, B*T, 19]

        a_pl_1 = a_pl_1.permute(1,0,2)  # [B*T,5,19]
        a_pl_2 = a_pl_2.permute(1,0,2)  # [B*T,5,19]
        
        action_cls_logits = self.ball_cls_head(decoder_out[1,...])  # [B*T, 2]

        
        #predict qsq
        offense_encode = state_feat[:,1:6,:,:]# [B,5,T,D]
        qsq = self.decoder_qsq(offense_encode) # [B,5,T,1]
        #print('qsq',qsq.shape)

        out = {
                "A": {"ball": {"1": a_ball_1, "2": a_ball_2},  # [B*T,1,C_ball]
                        "pl":   {"1": a_pl_1,   "2": a_pl_2},    # [B*T,5,C_pl]
                },
                "V": Value,        # [B,T,1]
                "qsq": qsq,      # [B,5,T,1]
                "action_cls": action_cls_logits,  # [B*T,2]
                "nbias": None
            }
            
        return out 

    # @torch.no_grad()
    # def get_random_action(self, batch_size):
    #     """
    #     第0个动作从 bin 0~5 中选，其余动作从 bin 6~24 中选。
    #     返回: [B, 6] 的 tensor, 每个位置是 bin 索引
    #     """
    #     device = self.device

    #     # 第0个位置：从 0~5 采样
    #     ball_action = torch.randint(0, 8, (batch_size, 1), device=device) #0-7

    #     # 第1~5个位置：从 6~24 采样
    #     player_actions = torch.randint(8, 27, (batch_size, 5), device=device) #8-26

    #     # 拼接得到完整 action tensor
    #     return torch.cat([ball_action, player_actions], dim=1)  # [B, 6]


    # @torch.no_grad() 
    # def get_optimal_actions(self, state_tokens, agent_ids, padding_mask, edge_index,edge_attr,prob_random_action=0.1):
    #     #
    #     """
    #         自回归推理，根据当前状态逐步预测动作序列。
        
    #     Args:
    #         state_tokens: [B*A, T, 1] B=1 ->[12,80,1]
    #         agent_ids: [B*A] -> [12]
    #         padding_mask: [A, T]-> [12,121]  
    #         prob_random_action: 随机探索概率(epsilon-greedy)
        
    #     Returns:
    #         optimal_actions: [B, num_agents, T]，每个agent每个时间步的动作索引
    #     """
    #     B = state_tokens.size(0) // 12  # 假设agent总数为12
    #     assert B == 1 
    #     T = state_tokens.size(1)
    #     device = state_tokens.device
    #     num_agents = 6  # 球和进攻球员数
    #     num_bins = self.cfg.num_bins

    #     optimal_actions = torch.zeros(B, num_agents, T, dtype=torch.long, device=device)

    #     with torch.no_grad():
    #         state_feat, player_embedding = self.state_embedding(state_tokens, padding_mask, agent_ids,edge_index,edge_attr)
    #         state_feat = rearrange(state_feat, '(b a) t d -> b a t d', a=12)
    #         player_embed = rearrange(player_embedding, '(b a) d -> b a d', a=12)
    #         player_embed_ = self.identify_embedding(player_embed[:, :6, :])

    #         for t in range(T):
    #             # 当前状态
    #             S = state_feat[:, :, t, :].mean(dim=1, keepdim=True)  # [B,1,D]
    #             P = player_embed_  # [B,6,D]

    #             # 已有动作作为输入
    #             seq = [S]
    #             for i in range(num_agents):
    #                 seq.append(P[:, i:i+1, :])
    #                 if t == 0:
    #                     seq.append(torch.zeros(B, 1, self.embed_dim, device=device))
    #                 else:
    #                     prev_action_embed = self.action_bin_embeddings(optimal_actions[:, i, t-1])
    #                     seq.append(prev_action_embed.unsqueeze(1))

    #             SPA = torch.cat(seq, dim=1)  # [B,13,D]

    #             decoder_input = rearrange(P, 'b a d -> a b d')
    #             prefix_memory = rearrange(SPA, 'b s d -> s b d')

    #             prefix_mask = build_prefix_mask(
    #                 T=num_agents,
    #                 prefix_lens=[2 * i + 1 for i in range(num_agents)],
    #                 total_mem_len=1 + 2 * num_agents,
    #                 device=device
    #             )

    #             causal_mask = torch.triu(torch.ones(num_agents, num_agents, device=device), diagonal=1).bool()

    #             decoder_out = self.decoder(decoder_input, prefix_memory, tgt_mask=causal_mask, memory_mask=prefix_mask)

    #             q1 = self.q_head_1(decoder_out)
    #             q2 = self.q_head_2(decoder_out)

    #             q_values = (q1 + q2) / 2  # [6,B,num_bins]

    #             optimal_actions_t = q_values.argmax(dim=-1).T

    #             if prob_random_action > 0:
    #                 random_actions = torch.randint(0, num_bins, optimal_actions_t.shape, device=device)
    #                 random_mask = torch.rand(optimal_actions_t.shape, device=device) < prob_random_action
    #                 optimal_actions_t = torch.where(random_mask, random_actions, optimal_actions_t)

    #             optimal_actions = optimal_actions.scatter(dim=2, index=torch.tensor([t], device=device).unsqueeze(0).unsqueeze(0), src=optimal_actions_t.unsqueeze(2))

    #     return optimal_actions  # [B, num_agents, T]
