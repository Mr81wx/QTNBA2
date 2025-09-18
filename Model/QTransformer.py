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


class Encoder(nn.Module):
    def __init__(
        self, device,
        in_feat_dim=1, time_steps=80,
        feature_dim=128, head_num=6,
        n_player_ids=501                 # 0 号留给球
    ):
        super().__init__()
        self.device = device
        self.T, self.D = time_steps, feature_dim
        self.A, self.H = 12, head_num
        assert feature_dim % head_num == 0

        #
        self.input_embedding = nn.Embedding(600,12)
        self.input_project = nn.Linear(4,12)

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

        self.layer_A = nn.Sequential(nn.Linear(12,32), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(32), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(32,128), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(128), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(128,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU() )
        self.layer_A.apply(init_xavier_glorot)

        # ---- PosEnc + Self‑Att ----
        self.posenc  = PositionalEncoding(feature_dim, 0.1, time_steps)
        self.selfatt1 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        self.selfatt2 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)
        self.selfatt3 = SelfAttLayerCausalTime(time_steps, feature_dim, head_num, 0.1)

        # ---- GATv2Conv（帧内）----
        self.gat1 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=1, concat=False)
        self.gat2 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=1, concat=False)
        self.gat3 = GATv2Conv(feature_dim, feature_dim, head_num, edge_dim=1, concat=False)


    # ---------------- forward ----------------
    def forward(self, state_feat, padding_mask, agent_ids,edge_index,edge_attr):
        """
        state_feat : (B*A, T, 5) [x,y,xv,yv,hex_index]
        padding_mask: (B*A, T)
        agent_ids  : (B*A,)
        edge_index: [B,T]
        edge_attr: [B,T,132,1]
        """

        N, T, _ = state_feat.shape
        B = N // self.A
        assert T == self.T

        #embedding states to 12
        #x,y,vx,vy
        state_con = state_feat[...,:4]
        state_xy = self.input_project(state_con)  # shape: [..., 12]
        #grid
        grid_id = state_feat[..., -1]
        grid_id  = grid_id.squeeze(-1)
        grid_id = torch.where(grid_id == 1e9, torch.full_like(grid_id, 599), grid_id) #实际有效值为0-503,padding 值 599,右边半场无效为598
        grid_id  = self.input_embedding(grid_id.long()) #[B*A,T,12]

        #qsq = state_feat[..., 1:]
        #qsq = self.qsq_project(qsq) #[B*A,T,12]

        # ---- Player Embedding ----
        emb = self.player_embedding(agent_ids.clamp(min=0))  # [B*A, 12]
        emb = emb.unsqueeze(1).expand(-1, T, -1)             # [N, T, 12]

        # ---- Index Embedding ----
        index_ids = torch.arange(self.A, device=self.device).view(1, self.A, 1).expand(B, self.A, T)  # [B, A, T]
        index_emb = self.index_embedding(index_ids).reshape(B * self.A, T, -1)    

        x = state_xy + grid_id + emb + index_emb #[B*A,T,12]

        # ---- Layer A ----
        #x = rearrange(x, 'n t d -> (n t) d')    # [N*T, D]
        x = self.layer_A(x)                     # e.g. Linear, MLP, LayerNorm...
        #x = rearrange(x, '(n t) d -> n t d', n=N, t=T)  # [N, T, D]

        player_ = x.clone()
        
        # ---- PosEnc & Self‑Att + 三次 GAT ----
        x = self.posenc(x)

        
        for self_att, gat in [(self.selfatt1, self.gat1),
                              (self.selfatt2, self.gat2),
                              (self.selfatt3, self.gat3)]:
            x = self_att(x, padding_mask)                    # [N, T, D]

            x_btad = x.view(B, self.A, T, self.D).permute(0, 2, 1, 3)  # [B, T, A, D]

            x_btad = run_gat_framewise_with_batch(
                x_btad, gat,
                edge_index,
                edge_attr
            )

            x = x_btad.permute(0, 2, 1, 3).reshape(N, T, self.D)  # [N, T, D]

        return x, player_
 


#decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

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
        x = x + self_attn_output

        # 2. Cross-attention to prefix memory
        x_norm = self.ln2(x)
        cross_attn_output, cross_attn_weights = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        x = x + cross_attn_output

        # 3. Feedforward
        x_norm = self.ln3(x)
        x = x + self.ff(x_norm)

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
        
        tgt_mask = tgt_mask.to(x.device)
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

        self.state_embedding = Encoder(self.device, in_feat_dim=cfg.in_feat_dim, time_steps=cfg.time_steps, feature_dim=self.embed_dim, head_num=cfg.encoder_n_head)
        self.identify_embedding = nn.Sequential(
                                    nn.Linear(12, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, self.embed_dim)
                                    )
        self.action_bin_embeddings = nn.Embedding(cfg.num_bins + 1 , cfg.embed_dim) #有一个padding

        self.decoder = Decoder(
            num_layers=cfg.decoder_layers,
            embed_dim=self.embed_dim,
            num_heads=cfg.n_head,
            dropout=cfg.dropout,
            max_seq_len=self.max_tokens
        )

        self.q_head_1 = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.embed_dim, cfg.num_bins_player)
        )

        self.q_head_2 = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.embed_dim, cfg.num_bins_player)
        )

        self.qball_head_1 = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.embed_dim, cfg.num_bins_ball)
        )

        self.qball_head_2 = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.embed_dim, cfg.num_bins_ball)
        )



        self.q_head_1.apply(init_weights)
        self.q_head_2.apply(init_weights)
        self.qball_head_1.apply(init_weights)
        self.qball_head_2.apply(init_weights)


    def forward(self, state_tokens, agent_ids, padding_mask, action_tokens,edge_index,edge_attr):
        '''state_tokens [B*A,T,1]
           agent_ids [B*A] 
           padding_mask [B*A,T]
           action_tokens [B*A,T]
           edge_index: [2,132]
           edge_attr: [B,T,132,1]
           '''
        
        
        
        # Encode Step
        state_feat,player_embedding = self.state_embedding(state_tokens, padding_mask, agent_ids,edge_index,edge_attr)
        #print("state_feat NaN:", torch.isnan(state_feat).any())
        state_feat = rearrange(state_feat, '(b a) t d -> b a t d', a=12)
        state_embed = state_feat.mean(dim=1) # [B, T, D]

        B, T, D = state_embed.shape
        
        #Decoder Step
       
        #player embedding [B,D]
        player_embed = rearrange(player_embedding, '(b a) t d -> b a t d', a=12)
        player_embed_ = player_embed[:,:6,:,:] #[B,6,T,D]
        #player_embed_ = self.identify_embedding(player_embed_) #[B,6,T,D]

        #player embedding [B,T,D] 直接用node feature 
        #player_embed_ = state_feat[:,:6,:,:] # [B,6,T,D] 5.25修改
        
        #action action_tokens [B,6,T,1]
        action_tokens = rearrange(action_tokens, '(b n) t -> b n t',n=6) # [B,6,T,1]
        action_tokens = action_tokens.clamp(min=0) #确保action 没有负数，0-7 为球，8-26
        action_embed = self.action_bin_embeddings(action_tokens) # [B,6,T,D]
        #action_embed_ = rearrange(action_embed, '(b n t) d -> b n t d', b = Batch_size,n=6,t=T)# B,6,T,D

        #bulid prefix_memory [B*T,S,D] Sequence = [State,P0 embedd, P0 action, P1 embedd, P1 action....]
        S = rearrange(state_embed, 'b t d -> (b t) d').unsqueeze(1)   # B*T,1,D
        P = rearrange(player_embed_, 'b a t d -> (b t) a d') # B*T,6,D
        A = rearrange(action_embed , 'b a t d -> (b t) a d') # B*T,6,D

        A = P + A # B*T,6,D

        seq = [S]  # 首先加上 state token

        # 按照 P0, A0, ..., P5, A5 拼接
        for i in range(6):
            seq.append(P[:, i:i+1, :])  # 取第 i 个 player embedding，形状 [B*T, 1, D]
            seq.append(A[:, i:i+1, :])  # 取第 i 个 action embedding，形状 [B*T, 1, D]

        # 拼接
        SPA = torch.cat(seq, dim=1)  # [B*T, 13, D]

        assert P.shape == (B*T, 6, D)
        assert SPA.shape == (B*T, 13, D)

        decoder_input  = rearrange(P, 'b a d -> a b d')     # [6, B*T, D]
        prefix_memory  = rearrange(SPA, 'b s d -> s b d')   # [13, B*T, D]

        prefix_mask = build_prefix_mask(
            T = self.num_actions,
            prefix_lens=[2 * i + 1 for i in range(self.num_actions)],
            total_mem_len=1 + 2 * self.num_actions,
            device=self.device
        )

        causal_mask = torch.triu(torch.ones(self.num_actions, self.num_actions, device=self.device), diagonal=1).bool()
        # Decode
        decoder_out = self.decoder(decoder_input, prefix_memory, tgt_mask=causal_mask, memory_mask=prefix_mask)

        # Predict Q-values at each decoder step
        q1_players = self.q_head_1(decoder_out[1:,:,:])   #[n b d]
        q2_players  = self.q_head_2(decoder_out[1:,:,:])

        q_min_players  = torch.min(q1_players, q2_players)
        q_avg_players  = (q1_players + q2_players) / 2

        q_min_players  = rearrange(q_min_players ,'n b d -> b n d')
        q_avg_players  = rearrange(q_avg_players ,'n b d -> b n d')
        q1_players  = rearrange(q1_players ,'n b d -> b n d')
        q2_players  = rearrange(q2_players ,'n b d -> b n d')

        q1_ball = self.qball_head_1(decoder_out[0:1,:,:])
        q2_ball = self.qball_head_2(decoder_out[0:1,:,:])

        q_min_ball = torch.min(q1_ball, q2_ball)
        q_avg_ball = (q1_ball+ q2_ball) / 2

        q_min_ball= rearrange(q_min_ball,'n b d -> b n d')
        q_avg_ball = rearrange(q_avg_ball,'n b d -> b n d')
        q1_ball = rearrange(q1_ball,'n b d -> b n d')
        q2_ball = rearrange(q2_ball,'n b d -> b n d')

        out =  {
        "players": {
            "min": q_min_players,
            "avg": q_avg_players,
            "1":   q1_players,
            "2":   q2_players,
        },
        "ball": {
            "min": q_min_ball,
            "avg": q_avg_ball,
            "1":   q1_ball,
            "2":   q2_ball,
        }
    }
        return out #[B*T,a, num_bins]

    @torch.no_grad()
    def get_random_action(self, batch_size):
        """
        第0个动作从 bin 0~5 中选，其余动作从 bin 6~24 中选。
        返回: [B, 6] 的 tensor, 每个位置是 bin 索引
        """
        device = self.device

        # 第0个位置：从 0~5 采样
        ball_action = torch.randint(0, 8, (batch_size, 1), device=device) #0-7

        # 第1~5个位置：从 6~24 采样
        player_actions = torch.randint(8, 27, (batch_size, 5), device=device) #8-26

        # 拼接得到完整 action tensor
        return torch.cat([ball_action, player_actions], dim=1)  # [B, 6]


    @torch.no_grad() 
    def get_optimal_actions(self, state_tokens, agent_ids, padding_mask, edge_index,edge_attr,prob_random_action=0.1):
        #
        """
            自回归推理，根据当前状态逐步预测动作序列。
        
        Args:
            state_tokens: [B*A, T, 1] B=1 ->[12,80,1]
            agent_ids: [B*A] -> [12]
            padding_mask: [A, T]-> [12,121]  
            prob_random_action: 随机探索概率(epsilon-greedy)
        
        Returns:
            optimal_actions: [B, num_agents, T]，每个agent每个时间步的动作索引
        """
        B = state_tokens.size(0) // 12  # 假设agent总数为12
        assert B == 1 
        T = state_tokens.size(1)
        device = state_tokens.device
        num_agents = 6  # 球和进攻球员数
        num_bins = self.cfg.num_bins

        optimal_actions = torch.zeros(B, num_agents, T, dtype=torch.long, device=device)

        with torch.no_grad():
            state_feat, player_embedding = self.state_embedding(state_tokens, padding_mask, agent_ids,edge_index,edge_attr)
            state_feat = rearrange(state_feat, '(b a) t d -> b a t d', a=12)
            player_embed = rearrange(player_embedding, '(b a) d -> b a d', a=12)
            player_embed_ = self.identify_embedding(player_embed[:, :6, :])

            for t in range(T):
                # 当前状态
                S = state_feat[:, :, t, :].mean(dim=1, keepdim=True)  # [B,1,D]
                P = player_embed_  # [B,6,D]

                # 已有动作作为输入
                seq = [S]
                for i in range(num_agents):
                    seq.append(P[:, i:i+1, :])
                    if t == 0:
                        seq.append(torch.zeros(B, 1, self.embed_dim, device=device))
                    else:
                        prev_action_embed = self.action_bin_embeddings(optimal_actions[:, i, t-1])
                        seq.append(prev_action_embed.unsqueeze(1))

                SPA = torch.cat(seq, dim=1)  # [B,13,D]

                decoder_input = rearrange(P, 'b a d -> a b d')
                prefix_memory = rearrange(SPA, 'b s d -> s b d')

                prefix_mask = build_prefix_mask(
                    T=num_agents,
                    prefix_lens=[2 * i + 1 for i in range(num_agents)],
                    total_mem_len=1 + 2 * num_agents,
                    device=device
                )

                causal_mask = torch.triu(torch.ones(num_agents, num_agents, device=device), diagonal=1).bool()

                decoder_out = self.decoder(decoder_input, prefix_memory, tgt_mask=causal_mask, memory_mask=prefix_mask)

                q1 = self.q_head_1(decoder_out)
                q2 = self.q_head_2(decoder_out)

                q_values = (q1 + q2) / 2  # [6,B,num_bins]

                optimal_actions_t = q_values.argmax(dim=-1).T

                if prob_random_action > 0:
                    random_actions = torch.randint(0, num_bins, optimal_actions_t.shape, device=device)
                    random_mask = torch.rand(optimal_actions_t.shape, device=device) < prob_random_action
                    optimal_actions_t = torch.where(random_mask, random_actions, optimal_actions_t)

                optimal_actions = optimal_actions.scatter(dim=2, index=torch.tensor([t], device=device).unsqueeze(0).unsqueeze(0), src=optimal_actions_t.unsqueeze(2))

        return optimal_actions  # [B, num_agents, T]
