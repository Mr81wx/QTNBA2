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
import torch_geometric
#from utils import init_weights, default, batch_select_indices, zero_
#from gpt_like import Transformer

from .utils import *
from torch_geometric.nn import GATv2Conv


def _init_linear_kaiming(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def run_gat_framewise_direct_batch(x_btad, gat_layer, edge_index, edge_attr):
    """
    直接批量处理版本：最高效
    前提：所有图有相同的边结构
    
    x_btad: [B, T, A, D]
    edge_index: [2, E]
    edge_attr: [B, T, E, F]
    
    """
    B, T, A, D = x_btad.shape
    _, E = edge_index.shape
    F = edge_attr.size(-1)
    
    # 1. 扁平化节点特征 [B*T*A, D]
    x_flat = x_btad.reshape(-1, D)  # [B*T*A, D]
    
    # 2. 为每个图创建偏移的边索引
    # 每个图的节点偏移量
    offsets = torch.arange(B * T, device=x_btad.device) * A  # [B*T]
    
    # 扩展边索引 [2, B*T*E]
    # 方法1：使用广播（内存高效）
    edge_index = torch.tensor(edge_index, device=x_btad.device)  # [2, E]
    edge_index_expanded = edge_index.unsqueeze(1) + offsets.view(1, -1, 1)  # [2, B*T, E]
    edge_index_flat = edge_index_expanded.reshape(2, -1)  # [2, B*T*E]
    
    # 3. 扁平化边特征 [B*T*E, F]
    edge_attr_flat = edge_attr.reshape(-1, F)  # [B*T*E, F]
    
    # 4. 运行GAT
    out = gat_layer(x_flat, edge_index_flat, edge_attr_flat)  # [B*T*A, D]
    
    # 5. 恢复形状
    return out.reshape(B, T, A, D)

class SelfAttLayerCausalTime(nn.Module):
    def __init__(self, time_steps=80, feature_dim=128, head_num=4, dropout=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num

        self.ln1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim,
                                          num_heads=head_num,
                                          dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

    def forward(self, x, padding_mask=None):
        """
        x: [B*A, T, D]
        padding_mask: [B*A, T] (True = pad)
        """
        BxA, T, D = x.shape
        assert T == self.time_steps

        # LayerNorm
        x_norm = self.ln1(x)

        # Correct causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        # Self-attention
        attn_output, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )

        # Residual 1
        x = x + attn_output

        # --- FFN block: Pre-LN ---
        x_norm = self.ln2(x)
        x = x + self.ffn(x_norm)

        return x


#1212 Encoder casusal att + GAT
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
        self.in_feat_dim = 4

        # ---- Embedding ----
        self.player_embedding = nn.Embedding(n_player_ids, 12)

        self.layer_A = nn.Sequential(nn.Linear(16,64), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(64), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(64,256), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(256), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(256,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU() )
        self.layer_A.apply(init_xavier_glorot)

        # ---- PosEnc + Self‑Att ----
        self.posenc  = PositionalEncoding(feature_dim, 0.1, time_steps)
        self.att1 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        self.att2 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        self.att3 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)

        # ---- GATv2Conv（帧内）----
        self.gat1 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=2, concat=False)
        self.gat2 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=2, concat=False)
        self.gat3 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=2, concat=False)

        self.norm1 = nn.LayerNorm(self.D)
        self.norm2 = nn.LayerNorm(self.D)
        self.norm3 = nn.LayerNorm(self.D)

    # ---------------- forward ----------------
    def forward(self, state_feat, padding_mask, agent_ids,edge_index,edge_attr):
        """
        state_feat : (B*A, T, 4) [x,y,vx,vy]
        padding_mask: (B*A, T) # if add spatial-attention
        agent_ids  : (B*A,)
        edge_index: [2,E]
        edge_attr: [B,T,132,2]
        """

        N, T, _ = state_feat.shape
        B = N // self.A
        assert T == self.T

        # ---- Player Embedding ----
        player_emb = self.player_embedding(agent_ids.clamp(min=0))  # [B*A, 12]
        player_emb = player_emb.unsqueeze(1).expand(-1, T, -1)             # [N, T, 12]

        x = torch.cat([state_feat, player_emb], dim=-1)  #[B*A,T,16]

        
        # ---- Layer A ----
        x = self.layer_A(x)      # [B*A, T, 16]                # e.g. Linear, MLP, LayerNorm...
      
        x_initial = x.clone() #[B*A, T, 16]  
        # ---- PosEnc & Self‑Att + 三次 GAT ----
        x = self.posenc(x) # [B*A, T, 16]   self attn 位置编码

        
        for  att, gat, norm_gat in zip(
            [self.att1,self.att2,self.att3],
            [self.gat1, self.gat2, self.gat3], 
            [self.norm1, self.norm2,self.norm3]):

            x = att(x, padding_mask)                    # [N, T, D]

            x_norm = norm_gat(x)

            x_btad = x_norm.view(B, self.A, T, self.D).permute(0, 2, 1, 3)  # [B, T, A, D]

            x_btad = run_gat_framewise_direct_batch(
                x_btad, gat,
                edge_index,
                edge_attr
            )

            x_gat = x_btad.permute(0, 2, 1, 3).reshape(N, T, self.D) # [N, T, D]

            x = x + x_gat   # Residual connection + LayerNorm

           

        return x_initial,x #[N, T, D]
 

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

        # 1. Causal self-attention
        x_norm = self.ln1(x)
        self_attn_output, self_attn_weights = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        #x = x + self_attn_output
        x = x + self.drop1(self_attn_output)

        # 2. Cross-attention to prefix memory
        x_norm = self.ln2(x)
        cross_attn_output, cross_attn_weights = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        #x = x + cross_attn_output

        x = x + self.drop2(cross_attn_output)

        # 3. Feedforward
        x_norm = self.ln3(x)
        ff_out = self.ff(x_norm)
        x = x + self.drop3(ff_out)

        if return_attn_weights:
            return x, self_attn_weights, cross_attn_weights
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
 
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



class MultiHeadBallPointer(nn.Module):
    """
    Multi-head pointer Q-head for ball actions, with explicit defense attention.

    输入:
        ball_h: [B, T, D]          每个时间步的 ball token 表示
        off_h:  [B, T, N_off, D]   每个时间步 N_off 个进攻球员 token 表示
        def_h:  [B, T, N_def, D]   每个时间步 N_def 个防守球员 token 表示
        rim_h:  [B, T, D]          每个时间步 rim token 表示.          

    输出:
        q_ball: [B, T, 1 + N_off]
            对动作 {shoot_rim(0), pass_to_off1(1), ..., pass_to_offN(1+N_off-1)} 的 Q 值
    """
    def __init__(
        self,
        d_model: int,
        num_offense: int = 5,
        num_defense: int = 5,
        num_heads_pointer: int = 4,
        use_defense_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_offense = num_offense
        self.num_defense = num_defense
        self.num_heads = num_heads_pointer
        assert d_model % num_heads_pointer == 0, "d_model 必须能被 num_heads_pointer 整除"
        self.head_dim = d_model // num_heads_pointer
        self.scale = self.head_dim ** 0.5

        self.use_defense_attention = use_defense_attention

        # --------- 1) Offense-Defense Attention（构造每个进攻球员的防守上下文）---------
        if use_defense_attention:
            # Q from offense, K/V from defense (单头即可，多头负担在 pointer 上)
            self.def_q_proj = nn.Linear(d_model, d_model)
            self.def_k_proj = nn.Linear(d_model, d_model)
            self.def_v_proj = nn.Linear(d_model, d_model)

            # 将 [off_h, def_context] 融合回 D 维
            self.off_def_fuse = nn.Linear(d_model * 2, d_model)

        # --------- 2) Pointer Multi-Head Q/K 投影 ---------
        self.q_proj = nn.Linear(d_model, d_model)  # for ball
        self.k_proj = nn.Linear(d_model, d_model)  # for memory targets (rim + offense)

    def forward(
        self,
        ball_h: torch.Tensor,    # [B, T, D]
        off_h: torch.Tensor,     # [B, T, N_off, D]
        def_h: torch.Tensor,     # [B, T, N_def, D]
        rim_h: torch.Tensor,     # [B, T, D]
        action_mask: torch.Tensor = None,  # [B, T, 1+N_off] (optional, True=mask)
    ) -> torch.Tensor:
        B, T, D = ball_h.shape
        _, _, N_off, D_off = off_h.shape
        _, _, N_def, D_def = def_h.shape

        assert D_off == D_def == D == self.d_model
        assert N_off == self.num_offense, "num_offense mismatch"
        assert N_def == self.num_defense, "num_defense mismatch"

        # ==============================
        # 1) Defense attention (可选)
        # ==============================
        if self.use_defense_attention:
            # flatten 时间+batch 维，方便做 attention
            off_flat = rearrange(off_h, 'b t n d -> (b t) n d')  # [B*T, N_off, D]
            def_flat = rearrange(def_h, 'b t n d -> (b t) n d')  # [B*T, N_def, D]

            # Q: offense, K/V: defense
            q_def = self.def_q_proj(off_flat)  # [B*T, N_off, D]
            k_def = self.def_k_proj(def_flat)  # [B*T, N_def, D]
            v_def = self.def_v_proj(def_flat)  # [B*T, N_def, D]

            # logits: [B*T, N_off, N_def]
            attn_logits = torch.matmul(
                q_def, k_def.transpose(-2, -1)
            ) / (D ** 0.5)

            attn_weights = torch.softmax(attn_logits, dim=-1)  # softmax over defenders

            # 防守上下文: [B*T, N_off, D]
            def_context_flat = torch.matmul(attn_weights, v_def)

            # reshape 回 [B, T, N_off, D]
            def_context = rearrange(def_context_flat, '(b t) n d -> b t n d', b=B, t=T)

            # offense + defense context 融合: [B, T, N_off, D]
            off_cat = torch.cat([off_h, def_context], dim=-1)      # [B,T,N_off,2D]
            off_enhanced = torch.tanh(self.off_def_fuse(off_cat))  # [B,T,N_off,D]
        else:
            off_enhanced = off_h  # 不用防守 attention 就直接用进攻 embedding

        # ==============================
        # 2) 构建 pointer memory: [rim, offense_enhanced]
        # ==============================
        rim_expanded = rim_h.unsqueeze(2)                   # [B, T, 1, D]
        mem = torch.cat([rim_expanded, off_enhanced], 2)    # [B, T, 1+N_off, D]
        N_mem = 1 + N_off

        # ==============================
        # 3) Multi-Head Pointer: Q(s, a) = dot(q(ball), k(target))
        # ==============================
        BT = B * T

        # ball: [B,T,D] -> [BT,D]
        ball_flat = rearrange(ball_h, 'b t d -> (b t) d')           # [BT, D]
        # mem:  [B,T,N,D] -> [BT,N,D]
        mem_flat  = rearrange(mem,    'b t n d -> (b t) n d')       # [BT, N_mem, D]

        # Q: [BT, H, Hd]
        q = self.q_proj(ball_flat)                                  # [BT, D]
        q = q.view(BT, self.num_heads, self.head_dim)               # [BT, H, Hd]

        # K: [BT, H, N_mem, Hd]
        k = self.k_proj(mem_flat)                                   # [BT, N_mem, D]
        k = k.view(BT, N_mem, self.num_heads, self.head_dim)        # [BT, N_mem, H, Hd]
        k = k.permute(0, 2, 1, 3)                                   # [BT, H, N_mem, Hd]

        # 计算多头 pointer scores: [BT, H, N_mem]
        q_exp = q.unsqueeze(2)                                      # [BT, H, 1, Hd]
        scores = torch.matmul(q_exp, k.transpose(-2, -1))           # [BT, H, 1, N_mem]
        scores = scores.squeeze(2) / self.scale                     # [BT, H, N_mem]

        # Aggregation over heads: 取平均（也可以用 sum）
        scores_agg = scores.mean(dim=1)                             # [BT, N_mem]

        # ==============================
        # 4) 可选 action_mask: mask 无效动作
        # ==============================
        if action_mask is not None:
            # action_mask: [B,T,N_mem], True 表示 mask
            mask_flat = rearrange(action_mask, 'b t n -> (b t) n')  # [BT, N_mem]
            scores_agg = scores_agg.masked_fill(mask_flat, float('-1e9'))

        # reshape 回 [B, T, N_mem]
        q_ball = rearrange(scores_agg, '(b t) n -> b t n', b=B, t=T)  # [B, T, 1+N_off]

        return q_ball




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

        self.num_agents = cfg.num_actions # decoder 序列 tokens 数 (6 = 1 ball + 5 players)
        self.num_bins = cfg.num_bins     # action bins per agent (36, padding 是40)
        self.embed_dim = cfg.embed_dim   # transformer embedding dim (128)
        #self.max_tokens = cfg.max_tokens # decoder 最大序列长度 (15)

        self.num_bins_player = cfg.num_bins_player # action bins for player (36)，padding = 40
        self.num_bins_ball = cfg.num_bins_ball # action bins for ball (6)，padding = 40

        self.decoder_layers = 2

        self.state_embedding = Encoder(self.device, in_feat_dim=cfg.in_feat_dim, time_steps=cfg.time_steps, feature_dim=self.embed_dim, head_num=cfg.encoder_n_head)
        
        self.ball_action_bin_embeddings = nn.Embedding(cfg.num_bins+1, cfg.embed_dim,padding_idx=40) #有一个padding
        self.player_action_bin_embeddings = nn.Embedding(cfg.num_bins+1, cfg.embed_dim,padding_idx=40) #有一个padding

        self.posenc  = PositionalEncoding(self.embed_dim, 0.1, 12)

        self.fuse_layer = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.decoder = Decoder(
            num_layers=self.decoder_layers,
            embed_dim=self.embed_dim,
            num_heads=cfg.n_head,
            dropout=cfg.dropout
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

        self.Value_ball_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, 1)
        )

        self.ball_pointer_head_1 = MultiHeadBallPointer(d_model=self.embed_dim)
        self.ball_pointer_head_2 = MultiHeadBallPointer(d_model=self.embed_dim)
        
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
        '''state_tokens [B*A,T,4]
           agent_ids [B*A] 
           padding_mask [B*A,T]
           action_tokens [B*A,T]
           edge_index: [2,132]
           edge_attr: [B,T,132,1]
           '''
        
        
        # Encode Step
        state_initial, state_feat = self.state_embedding(state_tokens, padding_mask, agent_ids,edge_index,edge_attr) #[B*A,T,D]
        # [B*A,T,D]
        # 获得最初的state embedding 和经过encoder的state feature
        
        state_initial = rearrange(state_initial, '(b a) t d -> b a t d', a=12)
        state_feat = rearrange(state_feat, '(b a) t d -> b a t d', a=12)

        state_embed = state_feat.mean(dim=1) # [B, T, D] for Value head


        # = rearrange(Value, 'b t d -> (b t) d')  # [B*T,1]
        # --Get player actions via decoder----------------
        B, T, D = state_embed.shape

        player_embed_ = state_initial[:,:6,:,:] #[B,6,T,D] 获得球+5个进攻球员的embedding
        
        #Decoder Step
        
        action_tokens_ = rearrange(action_tokens, '(b n) t -> b n t',n=6) # [B,6,T]
        ball_action_tokens = action_tokens_[:,0:1,:] #[B,1,T]
        player_action_tokens = action_tokens_[:,1:,:] #[B,5,T]

        ball_action_embed = self.ball_action_bin_embeddings(ball_action_tokens)  # [B, 1, T, D]
        player_action_embed = self.player_action_bin_embeddings(player_action_tokens)  # [B, 5, T, D]
        action_embed = torch.cat([ball_action_embed, player_action_embed], dim=1) # [B, 6, T, D]
        
        #S = rearrange(state_embed, 'b t d -> (b t) d')  # B*T,D 与
        
        P = rearrange(player_embed_, 'b a t d -> (b t) a d') # B*T,6,D # [ball  o1 o2 o3 o4 o5] player embedding
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
        #     T = self.num_agents,
        #     prefix_lens=[2 * i + 1 for i in range(self.num_agents)],
        #     total_mem_len=1 + 2 * self.num_agents,
        #     device=self.device
        # )
        
        #prefix_mask = torch.triu(torch.ones(12, 12, dtype=torch.bool, device=self.device), diagonal=1)
        prefix_mask = torch.zeros(12, 12, dtype=torch.bool, device=self.device)

        causal_mask = torch.triu(torch.ones(self.num_agents *2 , self.num_agents *2, device=self.device), diagonal=1).bool()
        # Decode
        decoder_out = self.decoder(decoder_input, prefix_memory, tgt_mask=causal_mask, memory_mask=prefix_mask)
        #decoder_out [12, B*T, D] decoder 只负责球员的动作预测

        decoder_out_ = decoder_out[0::2, ...]  # [6, B*T, D]
        decoder_out_player = decoder_out_[1:6, ...]  # [5, B*T, D]
        a_pl_1 = self.A_pl_head_1(decoder_out_player)  # [5, B*T, 19]
        a_pl_2 = self.A_pl_head_2(decoder_out_player)  # [5, B*T, 19]

        a_pl_1 = a_pl_1.permute(1,0,2)  # [B*T,5,19]
        a_pl_2 = a_pl_2.permute(1,0,2)  # [B*T,5,19]

        # --Get Value-----------
        #s_for_V = decoder_out[1::2,...]  # [6, B*T, D]
        s_pl = decoder_out_player  # [5, B*T, D]
        s_pl = rearrange(s_pl,'a (b t) d -> b t a d',b=B)#s_pl.permute(1,0,2).view(B, T, 5, D)  # [B,T,5,D]
        V_pl   = self.Value_head(s_pl)  #[B,T,5,1]
        


        # --Get ball action pointer Q-values-----
        ball_h = state_feat[:,0,...]   # [B,T,D]
        #ball_h = decoder_out[0,...]  # [B*T,D]
        #ball_h = rearrange(ball_h,'(b t) d -> b t d',b=B)  # [B,T,D]

        off_h = state_feat[:,1:6,...].permute(0,2,1,3)  #  [B,T,5,D]
        def_h = state_feat[:,6:11,...].permute(0,2,1,3) # [B,T,5,D]
        rim_h = state_feat[:,-1,...]  # [B,T,D]

        a_ball_1 = self.ball_pointer_head_1(
           ball_h, off_h, def_h, rim_h 
        )  # [B,T, 6]

        a_ball_2 = self.ball_pointer_head_2(
           ball_h, off_h, def_h, rim_h 
        )  # [B,T, 6]

        a_ball_1 = rearrange(a_ball_1,'b t d -> (b t) 1 d') # [B*T,1,6]
        a_ball_2 = rearrange(a_ball_2,'b t d -> (b t) 1 d') # [B*T,1,6]
        #print('a_ball_1',a_ball_1.shape)
        
        V_ball = self.Value_ball_head(ball_h)  
        #V_ball = self.Value_head(s_ball) #[B,T,1]
        
        
        
        #---predict action cls (dribble / not dribble)----
        action_cls_logits = self.ball_cls_head(decoder_out[1,...])  # [B*T, 2]

        
        #----predict qsq ---------
        offense_encode = state_feat[:,1:6,:,:]# [B,5,T,D]
        qsq = self.decoder_qsq(offense_encode) # [B,5,T,1]
        #print('qsq',qsq.shape)

        out = {
                "A": {"ball": {"1": a_ball_1, "2": a_ball_2},  # [B*T,1,C_ball]
                        "pl":   {"1": a_pl_1,   "2": a_pl_2},    # [B*T,5,C_pl]
                },
                "V": {"ball": V_ball,   # [B,T,1]
                      "pl":   V_pl    # [B,T,5,1]
                },        # [B,T,1]
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
