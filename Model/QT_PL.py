# 1212 no share V
import pytorch_lightning as pl
import copy
import torch
import torch.nn as nn
from ema_pytorch import EMA
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR


import copy, torch, torch.nn.functional as F
from einops import rearrange, repeat



from .QTransformer import QTransformer

from .utils import *


class QT(pl.LightningModule):
    def __init__(self, cfg):
        super(QT, self).__init__()

        self.cfg = cfg

        self.min_reward = cfg.min_reward
        self.discount_factor_gamma = cfg.discount_factor_gamma

        self.model = QTransformer(self.cfg.qtransformer)

        self.ema_model = EMA(self.model,include_online_model = False,
                             beta = cfg.ema.beta,
                             update_after_step = cfg.ema.update_after_step,
                             update_every = cfg.ema.update_every)
        
        self.soft_alpha = cfg.soft_alpha
        self.only_ball = cfg.only_ball 
    
    def forward(self, batch):
        state_tokens = batch['state_tokens'] # [B*A,T,4]    
        agent_ids = batch['agent_ids']
        padding_mask = batch['padding_mask'] #[B*A,T] 
        action_tokens = batch['action_tokens'] #  #40为无效帧补齐的

        edge_index = batch['edge_index']  #[2,110]
        edge_attr = batch['new_edge_attr'] #[B,T,110,2]

        rewards = batch['rewards'] #[B,T] 实际最后一个timesteps都是shot，只有a0一个action -1e9 为无效
        done = batch['done'] # [B] 每一个回合的实际长度

        #monte_carlo_return = default(batch['mc_return'], -1e4) #[B,T]

        out = self.model(state_tokens, agent_ids, padding_mask, action_tokens, edge_index, edge_attr) 
        # out =  {
        # "A": heads["A"],   # # [B*T,1,C_ball] [B*T,5,C_pl]
        # "V": heads["V"],     # [B*T,1,1] [B*T,5,1]   # 现在是四头结构：ball.{1,2}, pl.{1,2}
        # "nbias": heads["nbias"], 
        # "qsq": qsq # [B,6,T,1]
        #}

        return out

    def training_step(self, batch, batch_idx):
        total_loss,td_loss,cql_loss,qsq_loss,ce_loss = self.compute_loss(batch)

        self.log_dict({
            "train/total_loss": total_loss,
            "train/td_loss": td_loss,
            "train/cql_loss": cql_loss,
            "train/qsq":qsq_loss,
            "train/ce_loss": ce_loss
            #"train/reverse_penalty": reverse_penalty,
        }, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False)
            
        return total_loss
    
    def training_step_end(self, outputs):
        
        self.ema_model.update() 
        return outputs

    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        # ---------- unpack ----------
        state_tokens  = batch['state_tokens']
        agent_ids     = batch['agent_ids']
        padding_mask  = batch['padding_mask']        # [B*A, T]
        action_tokens = batch['action_tokens']       # [B*A, T]
        rewards       = batch['rewards']             # [B, T]
        done          = batch['done']                # [B]
        edge_index    = batch['edge_index']
        edge_attr     = batch['edge_attr']
        #mc_return     = default(batch['mc_return'], -1e4) #[B, T]

        B, T = rewards.shape

        
        # ---------- forward ----------
        out = self.model(state_tokens, agent_ids, padding_mask, action_tokens, edge_index, edge_attr)

        #  out = {
        #         "A": {"ball": {"1": a_ball_1, "2": a_ball_2},  # [B*T,1,C_ball]
        #                 "pl":   {"1": a_pl_1,   "2": a_pl_2},    # [B*T,5,C_pl]
        #         },
        #         "V": {"ball": V_ball,   # [B,T,1]
        #               "pl":   V_pl    # [B,T,5,1]
        #         },        # [B,T,1]
        #         "qsq": qsq,      # [B,5,T,1]
        #         "action_cls": action_cls_logits,  # [B*T,2]
        #         "nbias": None
        #     }
        
        # ---------- avg-head logits ----------
        A_ball_1, A_ball_2 = out["A"]["ball"]["1"], out["A"]["ball"]["2"]
        A_ball_1 = A_ball_1 - A_ball_1.mean(dim=-1, keepdim=True) # [B*T,1,C_ball]
        A_ball_2 = A_ball_2 - A_ball_2.mean(dim=-1, keepdim=True)

            
        A_pl_1,   A_pl_2   = out["A"]["pl"]["1"], out["A"]["pl"]["2"]
        A_pl_1 = A_pl_1 - A_pl_1.mean(dim=-1, keepdim=True)   # [B*T,5,C_pl]
        A_pl_2 = A_pl_2 - A_pl_2.mean(dim=-1, keepdim=True)


        V = out["V"]
        V_ball = V['ball']  # [B,T,1]
        
        V_pl = V['pl']      # [B,T,5,1]

        V_ball = rearrange(V_ball,'b t d -> (b t) 1 d')  # [B*T,1,1]
        V_pl = rearrange(V_pl,'b t n d -> (b t) n d')  # [B*T,5,1]
        #V_pl = V_ball.expand(-1,5,-1)  # [B*T,5,1]
        nb       = out["nbias"] if out["nbias"] is not None else 0.0
        P = 5                     # 动态拿玩家行数
        A = 6                                    # 总行数 (ball + players)

        #print('V_pl',V_pl.shape,A_pl_1.shape)
        q_ball_avg = 0.5 * ((V_ball + nb + A_ball_1) + (V_ball + nb + A_ball_2))  # [B*T,1,C_ball]
        q_player_avg   = 0.5 * ((V_pl + nb + A_pl_1)   + (V_pl + nb + A_pl_2))    # [B*T,5,C_pl]

        #print('q_player_avg',q_player_avg.shape)

        # === 真实动作 ===
        ball_action = action_tokens[0::A, :]  # [B, T]
        player_action = rearrange(action_tokens, '(b a) t -> b t a', a=6)[..., 1:]  # [B, T, 5]
        player_action = rearrange(player_action, 'b t a -> (b t) a')  # [B*T, 5]

        # ---------- 1. Q mean ----------
        q_ball_selected = batch_ball_select_indices(q_ball_avg, action_tokens,6) # [B*T, 1]

        q_player_selected = batch_player_select_indices(q_player_avg , action_tokens) #[B*T, 5]

        #mask_ball = ((ball_action != 27) & (ball_action != 7)).reshape(-1) # [B*T]
        mask_ball = (ball_action != 40).reshape(-1) # [B*T]
        mask_ball = mask_ball.unsqueeze(-1)  # [B*T, 1]
        mask_player =  (player_action != 40) # [B*T,5]

        q_mean_ball = q_ball_selected[mask_ball.bool()].mean()
        q_mean_player = q_player_selected[mask_player.bool()].mean()
        
        #q_mean = 0.5 * q_mean_ball + 0.5 * q_mean_player
        self.log("val/q_mean_ball", q_mean_ball)
        self.log("val/q_mean_player", q_mean_player)

        # ---------- 2. policy accuracy ----------

      

        # === argmax 动作 ===
        q_ball_argmax = q_ball_avg.argmax(dim=-1)  # [B*T, 1]
        q_player_argmax = q_player_avg.argmax(dim=-1)  # [B*T, 5]
        q_player_argmax = q_player_argmax  # shift by 6 to match action space

        # reshape ball_action: [B, T] → [B*T, 1]
        ball_action_flat = ball_action.reshape(-1, 1)          # [B*T, 1]

        # === accuracy ===
        acc_ball = (q_ball_argmax == ball_action_flat)[mask_ball].float().mean()
        acc_player = (q_player_argmax == player_action)[mask_player].float().mean()

        self.log("val/acc_ball", acc_ball)
        self.log("val/acc_player", acc_player)
        

        # 1️⃣ flatten
        q_ball_argmax_flat = q_ball_argmax.squeeze(-1)        # [B*T]
        mask_ball_flat = mask_ball.squeeze(-1)                # [B*T], bool

        # 2️⃣ 只保留 valid 的 timestep
        q_ball_argmax_valid = q_ball_argmax_flat[mask_ball_flat]  # [N_valid]

        # 3️⃣ 统计每个 bin 的出现次数
        num_bins = q_ball_avg.shape[-1]   # e.g. 6
        bin_counts = torch.bincount(
            q_ball_argmax_valid,
            minlength=num_bins
        )  # [num_bins]

        # 4️⃣ 频率（可选）
        bin_freq = bin_counts.float() / bin_counts.sum().clamp_min(1)

        # 5️⃣ 打印
        #print("Q-ball argmax bin counts (valid only):", bin_counts.cpu().tolist())
        #print("Q-ball argmax bin frequency:", bin_freq.cpu().tolist())

        self.logger.experiment.add_histogram(
            tag="val/q_ball_argmax_valid",
            values=q_ball_argmax_valid,
            global_step=self.global_step
        )

        # 1️⃣ flatten
        q_player_argmax_flat = q_player_argmax.reshape(-1)     # [(B*T*5)]
        mask_player_flat = mask_player.reshape(-1)             # [(B*T*5)]

        # 2️⃣ 只保留 valid player-timestep
        q_player_argmax_valid = q_player_argmax_flat[mask_player_flat]  # [N_valid]

        # 3️⃣ bincount
        num_bins = q_player_avg.shape[-1]   # e.g. 19
        bin_counts = torch.bincount(
            q_player_argmax_valid,
            minlength=num_bins
        )

        bin_freq = bin_counts.float() / bin_counts.sum().clamp_min(1)

        # 4️⃣ 打印
        #print("Q-player argmax bin counts (valid only):", bin_counts.cpu().tolist())
        #print("Q-player argmax bin frequency:", bin_freq.cpu().tolist())

        self.logger.experiment.add_histogram(
            tag="val/q_player_argmax_valid",
            values=q_player_argmax_valid,
            global_step=self.global_step
        )

        #——-----------
        V_ball_flat = V_ball.squeeze(-1).squeeze(-1)   # [B*T]
        V_pl_flat   = V_pl.squeeze(-1)                 # [B*T, 5]
        B = done.size(0)
        T = V_ball_flat.numel() // B

        valid_mask_bt = build_loss_mask_from_lengths(done, T)  # [B, T], True=valid
        valid_mask_bt = rearrange(valid_mask_bt, 'b t -> (b t)').bool()  # [B*T]

        V_ball_valid = V_ball_flat[valid_mask_bt]   # [N_valid]
        valid_mask_pl = valid_mask_bt.unsqueeze(-1).expand_as(V_pl_flat)  # [B*T,5]
        V_pl_valid = V_pl_flat[valid_mask_pl]   # [N_valid * 5]

        self.logger.experiment.add_histogram(
            tag="val/V_ball_valid",
            values=V_ball_valid,
            global_step=self.global_step
        )

        self.logger.experiment.add_histogram(
            tag="val/V_player_valid",
            values=V_pl_valid,
            global_step=self.global_step
        )


    def compute_loss(self, batch):
        """
        split ball and player compute_loss
        """
        # ---------- 取 batch ----------
        state_tokens  = batch['state_tokens']     # [B*A, T, 1]
        agent_ids     = batch['agent_ids']        # [B*A]
        padding_mask  = batch['padding_mask']     # [B*A, T]
        action_tokens = batch['action_tokens']    # [B*6, T]
        action_tokens_cql = copy.deepcopy(action_tokens)

        #rewards  = batch['rewards']               # [B, T]
        rewards  = batch['rewards']      # [B, T]
        #done     = batch['done']                  # [B]
        done = batch['done'].clamp(max=80)
        edge_idx = batch['edge_index']            # [2,132]
        edge_attr= batch['edge_attr']             # [B,T,132,3]

        qsq_gt = batch['qsq']                     # [B*5,T]

        R_SCALE = 8.8665
        rewards = rewards / R_SCALE
        rewards = torch.tanh(rewards)
        

        B, T = rewards.shape
        #
        Y = self.discount_factor_gamma

        # #state mask， 对于在state中有球员在后场的情况都不去计算该球员的移动loss，{TODO_ 同时对于球的行为7的不去计算这一帧与球相关的loss}
        # state_x = state_tokens[...,0] #[B*A,T]
        # state_x = rearrange(state_x,'(b a) t -> b t a',b=B)
        # state_x = state_x[:,:,:6] # [B,T,5]
        # #invalid_state_mask = (state_x > 53).float()  # [B, T， 5]，
        # invalid_state_mask = (state_x > 53).any(dim=-1)  # [B, T] 对任意state中有进攻球员在后场，都直接不算其loss
        
        #调整 ball_action , 把所用运球的动作都改为运球动作 6
        # ball_action = action_tokens[0::6,:] #[B,T]  action_tokens [B*6,T]
        # ball_action_prev = torch.cat([ball_action[:, :1], ball_action[:, :-1]], dim=1) #[B,T]
        # valid_mask = (ball_action != 40) #[B,T] 40为无效帧补齐的
        # dribble = (ball_action == ball_action_prev) & valid_mask #true 表示这个动作是运球，False 表示这个位置的实际动作是传球 [B,T]
        # ball_action = torch.where(dribble==1, 6, ball_action)  #把所有运球动作都改为6
        # action_tokens[0::6, :] = ball_action


        # ---------- 前向 ----------
        # self.model 输出嵌套字典，两套 head
        out_pred = self.model(state_tokens, agent_ids,
                            padding_mask, action_tokens,
                            edge_idx, edge_attr)

        # out = {
        #         "A": {'pl': {'1': ..., '2': ...}, 'ball': {'1': ..., '2': ...}},  # [B*T,5,C_pl] [B*T,1,C_ball]
        #         "V": heads["V"],        # 现在是四头结构：ball.{1,2}, pl.{1,2}
        #         "nbias": heads["nbias"],
        #         "qsq": qsq
        #     }

        # players: [ B*T,5, 19]    ball: [ B*T,1, 8]
        a_pl_1  = out_pred['A']['pl']['1']  #[ B*T,5, 19] 
        a_pl_2  = out_pred['A']['pl']['2']  #[ B*T,5, 19] 
        
        a_ball_1 = out_pred['A']['ball']['1'] #[ B*T,1, 6]  11.12 dribble变为6 [B*T,1,7]
        a_ball_2 = out_pred['A']['ball']['2']

        # # ✅ 在这里加 penalty（推荐）
        # penalty = 0.5
        # a_ball_1[..., 0] -= penalty
        # a_ball_2[..., 0] -= penalty

        #v_pred = out_pred['V'] #[ B*T,1] 
        #v_pred = v_pred.unsqueeze(-1)#[ B*T,1, 1]

        V = out_pred["V"]
        V_ball = V['ball']  # [B,T,1]
        V_pl = V['pl']      # [B,T,5,1]

        V_ball = rearrange(V_ball,'b t d -> (b t) 1 d')  # [B*T,1,1]
        V_pl = rearrange(V_pl,'b t n d -> (b t) n d')  # [B*T,5,1]

        qsq_pred = out_pred['qsq'].squeeze(-1) # [B,5,T]
        action_logits = out_pred["action_cls"] # [B*T, 2]
        #print("qsq_pred:", qsq_pred.shape)

        # ---------- select 本步 Q ----------
        if out_pred['nbias'] is None:
            nb = 0.0

        alpha = self.soft_alpha

        #soft_a_ball_1 = a_ball_1 - alpha * torch.logsumexp(a_ball_1 / alpha, dim=-1, keepdim=True)
        #soft_a_ball_2 = a_ball_2 - alpha * torch.logsumexp(a_ball_2 / alpha, dim=-1, keepdim=True)

        #soft_a_pl_1 = a_pl_1 - alpha * torch.logsumexp(a_pl_1 / alpha, dim=-1, keepdim=True)
        #soft_a_pl_2 = a_pl_2 - alpha * torch.logsumexp(a_pl_2 / alpha, dim=-1, keepdim=True)

        a_ball_1 = a_ball_1 - a_ball_1.mean(dim=-1, keepdim=True) #[ B*T,1, 6] 
        a_ball_2 = a_ball_2 - a_ball_2.mean(dim=-1, keepdim=True) #[ B*T,1, 6] 

        a_pl_1 = a_pl_1 - a_pl_1.mean(dim=-1, keepdim=True) #[ B*T,5, 19] 
        a_pl_2 = a_pl_2 - a_pl_2.mean(dim=-1, keepdim=True) #[ B*T,5, 19] 

        adv1a_ball = batch_ball_select_indices( a_ball_1 ,action_tokens,6) #[B*T, 1] 
        adv2a_ball = batch_ball_select_indices( a_ball_2 ,action_tokens,6) #[B*T, 1]
    
        adv1a_pl = batch_player_select_indices( a_pl_1,action_tokens) #[B*T, 5]
        adv2a_pl = batch_player_select_indices( a_pl_2,action_tokens) #[B*T, 5]

        q_pred_1_ball = V_ball + adv1a_ball.unsqueeze(-1)          # [B*T,1,1]
        q_pred_2_ball = V_ball + adv2a_ball.unsqueeze(-1)         # [B*T,1,1]

        q_pred_1_player =  V_pl  + adv1a_pl.unsqueeze(-1)    # [B*T,5,1]
        q_pred_2_player =  V_pl  + adv2a_pl.unsqueeze(-1)


        q_pred_1 = torch.cat([q_pred_1_ball, q_pred_1_player], dim=1)
        q_pred_2 = torch.cat([q_pred_2_ball, q_pred_2_player], dim=1)

        q_pred_1 = rearrange(q_pred_1,'(b t) n 1-> b t n', b=B)
        q_pred_2 = rearrange(q_pred_2,'(b t) n 1-> b t n', b=B)
       
        # ---------- target 网络 ----------
        ball_action = action_tokens[0::6,:] #[B,T]  action_tokens [B*6,T]
        valid_mask_ball =  (ball_action != 40) #[B,T] 40为无效帧补齐的
        valid_mask_ball = rearrange(valid_mask_ball, 'b t -> (b t) 1')  #[B*T,1]

        player_action = rearrange(action_tokens,'(b a) t -> b a t',a = 6)[:,1:,:]
        player_action = rearrange(player_action, 'b a t -> (b t) a')  #[B*T,5]
        valid_mask_player = (player_action != 40) #[B*T,5]

        with torch.no_grad():
            out_target = self.ema_model(state_tokens, agent_ids,
                                        padding_mask, action_tokens,
                                        edge_idx, edge_attr)

            V_target = out_target['V'] 
            V_ball_target = V_target['ball']  # [B,T,1]
            V_pl_target = V_target['pl']      # [B,T,5,1]

            V_ball_target = rearrange(V_ball_target,'b t d -> (b t) 1 d')  # [B*T,1,1]
            V_pl_target = rearrange(V_pl_target,'b t n d -> (b t) n d')  # [B*T,5,1]
           

            a_target_pl_1  = out_target['A']['pl']['1']  #[ B*T,5, 19] 
            a_target_pl_2  = out_target['A']['pl']['2']  

            a_target_pl_1 = a_target_pl_1 - a_target_pl_1.mean(dim=-1, keepdim=True)
            a_target_pl_2 = a_target_pl_2 - a_target_pl_2.mean(dim=-1, keepdim=True)


            #soft_a_target_pl_1 = a_target_pl_1 - alpha * torch.logsumexp(a_target_pl_1 / alpha, dim=-1, keepdim=True) #[ B*T,5, 19] 
            #soft_a_target_pl_2 = a_target_pl_1 - alpha * torch.logsumexp(a_target_pl_2 / alpha, dim=-1, keepdim=True) #[ B*T,5, 19] 
            
            a_target_ball_1 = out_target['A']['ball']['1'] #[ B*T,1, 6]
            a_target_ball_2 = out_target['A']['ball']['2']
            a_target_ball_1 = a_target_ball_1 - a_target_ball_1.mean(dim=-1, keepdim=True)
            a_target_ball_2 = a_target_ball_2 - a_target_ball_2.mean(dim=-1, keepdim=True)  

            #11.16 
            # mask = torch.zeros(B*T, 7, device=a_target_ball_1.device, dtype=torch.float32)# [B*T]

            # holder = ball_action_prev.reshape(-1)   # [B*T] 
            # padding_mask = (holder == 40)                      # [B*T]
            # mask[padding_mask, :] = True

            # valid_holder = (holder >= 1) & (holder <= 5) 
            # idx = torch.arange(B*T, device=a_target_ball_1.device)[valid_holder]
            # mask[idx, holder[valid_holder]] = True
            # mask_ = mask.reshape(B*T, 1, 7).bool()  # [B*T,1,7]

            #a_target_ball_1 = a_target_ball_1.masked_fill(mask_, -1e3)
            #a_target_ball_2 = a_target_ball_1.masked_fill(mask_, -1e3)

           #-------------------------------------------
            q_pl_target_1 = V_pl_target + a_target_pl_1   # [B*T,5,19]
            q_pl_target_2 = V_pl_target + a_target_pl_2
            q_pl_target_max_1 = q_pl_target_1.max(dim=-1, keepdim=True).values       # [B*T,5,1]
            q_pl_target_max_2 = q_pl_target_2.max(dim=-1, keepdim=True).values

            q_pl_target   = torch.min(q_pl_target_max_1,q_pl_target_max_2)  # 取 avg 头 or min [ B*T,5, 1] 
            
            #q_ball_target_1 = V_target + soft_a_target_ball_1  #[ B*T,1, 1]
            #q_ball_target_2 = V_target + soft_a_target_ball_2
            #print("q_ball_target_1 before logsumexp:", q_ball_target_1.shape)
            #q_ball_target_1_ = alpha * torch.logsumexp(q_ball_target_1 / alpha, dim=-1, keepdim=True)  #[ B*T,1, 1]
            #q_balL_target_2_ = alpha * torch.logsumexp(q_ball_target_2 / alpha, dim=-1, keepdim=True)  #[ B*T,1, 1]
            #print("q_ball_target_1 after logsumexp:", q_ball_target_1.shape)
            q_ball_target_1 = V_ball_target + a_target_ball_1    # [B*T,1,6]
            q_ball_target_2 = V_ball_target + a_target_ball_2    # [B*T,1,6]

            # penalty = 0.1 #0102
            # q_ball_target_1[...,0] -= penalty  # shot 0102
            # q_ball_target_2[...,0] -= penalty  # shot 0102
            # max over action bins
            q_ball_target_max_1 = q_ball_target_1.max(dim=-1, keepdim=True).values   # [B*T,1,1]
            q_ball_target_max_2 = q_ball_target_2.max(dim=-1, keepdim=True).values

            # Double-Q: use min head
            q_ball_target = torch.min(q_ball_target_max_1, q_ball_target_max_2)      # [B*T,1,1]

            q_pl_target = q_pl_target.squeeze(-1)  # [B*T,5]
            q_ball_target = q_ball_target.squeeze(-1)  # [B*T,1]
           
            q_target = torch.cat([q_ball_target, q_pl_target], dim=-1) #[B*T, 6]
            q_target = rearrange(q_target, '(b t) n -> b t n', b=B) # [B, T, 6]


        # ---------- 分离predict [ball,p1,p2,p3,p4] [p5]   ----------
        # ---------- 分离target  [p1,p2,p3,p4,p5]   [ball] ----------
        done_mask = build_loss_mask_from_lengths(done, T).to(rewards.device) #[B,T] True for valid steps
        q_target_ball_next = q_target[...,0] #[B,T]
        rewards = rewards.clamp_min(-1e3)
        td_target_ball = rewards[:,:-1] + Y * (q_target_ball_next[:,1:] * done_mask[:,1:].float())
        # valid_counts = done_mask.sum(dim=-1)
        # print(valid_counts)
        # print(done)
        #print("td_target_ball:", td_target_ball.min())
        patched_q_target = q_target.clone()   # 浅 copy, 不污染原来的
        patched_q_target[:,1:,0] = td_target_ball

        # ---------- TD Loss  ---------- ball + player
        if self.only_ball:
            #只计算球的loss
            #只考虑球的动作，不考虑球员动作
            q_pred_first_1 = q_pred_1[...,0] #[B,T]
            q_pred_first_2 = q_pred_2[...,0] #[B,T]
            
            q_pred_first_1 = q_pred_first_1[...,:-1] #[B,T-1]
            q_pred_first_2 = q_pred_first_2[...,:-1] #[B,T-1]

            loss_ball_1 = F.smooth_l1_loss(q_pred_first_1, td_target_ball,reduction='none') #[B,T-1]
            loss_ball_2 = F.smooth_l1_loss(q_pred_first_2, td_target_ball,reduction='none') #[B,T-1]
            
            loss_ball_mask = done_mask[:,:-1] # [B,T-1] # loss mask

            loss_ball_1 = loss_ball_1 * pass_mask.float()#[B,T-1]
            loss_ball_2 = loss_ball_2 * pass_mask.float()#[B,T-1]

            td_loss_ball = loss_ball_1.mean() + loss_ball_2.mean()

            td_loss = td_loss_ball # td loss for pass and shot

        else:
            # === Flatten ===
            q_pred_1_flat = rearrange(q_pred_1, 'b t n -> b (t n)')
            q_pred_2_flat = rearrange(q_pred_2, 'b t n -> b (t n)')
            q_target_flat = rearrange(patched_q_target, 'b t n -> b (t n)')

            # === Step 1: TD elementwise loss ===
            loss_1 = F.smooth_l1_loss(q_pred_1_flat[:, :-1], q_target_flat[:, 1:], reduction='none')  # [B, T*6-1]
            length_flat = loss_1.size(1)
            loss_2 = F.smooth_l1_loss(q_pred_2_flat[:, :-1], q_target_flat[:, 1:], reduction='none')

            pad_col = torch.zeros((loss_1.size(0), 1), device=loss_1.device, dtype=loss_1.dtype)
            loss_1 = torch.cat([loss_1, pad_col], dim=1)  # [B, T*6]
            loss_2 = torch.cat([loss_2, pad_col], dim=1)  # [B, T*6]

            # === Step 2: 分离 ball / player ===
            ball_loss_1 = loss_1[:, 0::6]  # [B, T]
            ball_loss_2 = loss_2[:, 0::6]

            player_loss_1 = torch.stack(
                [loss_1[:, i::6] for i in range(1, 6)], dim=-1
            )  # [B, T, 5]
            player_loss_2 = torch.stack(
                [loss_2[:, i::6] for i in range(1, 6)], dim=-1
            )  # [B, T, 5]
            player_loss_1 = player_loss_1[:, :-1, :]  # [B, T-1, 5]
            player_loss_2 = player_loss_2[:, :-1, :]  # [B, T-1, 5]
            #print('player_loss_1',player_loss_1.shape)
            
            # === Step 3: 权重 (只给 ball) ===
            #weights = torch.where(dribble, 0.25, 15.0).float()  # [B, T] 
            weights = torch.ones_like(rewards)  # [B, T]
            ball_done_mask = build_loss_mask_from_lengths(done, T).to(rewards.device)
           
            td_loss_ball_1 = ball_loss_1 * weights * ball_done_mask.float()
            td_loss_ball_2 = ball_loss_2 * weights * ball_done_mask.float()

            # === Step 4: Masking ===
            player_done_mask = build_loss_mask_from_lengths(done-1, T-1).to(rewards.device) # [B, T-1] #loss mask
            player_done_mask = player_done_mask.unsqueeze(-1).expand(-1, -1, 5)  # [B, T, 5]
            td_player_loss_1 = player_loss_1  * player_done_mask.float()
            td_player_loss_2 = player_loss_2  * player_done_mask.float()
            # === Step 6: 最终TD损失 ===
            td_loss = td_loss_ball_1.mean() + td_loss_ball_2.mean() + td_player_loss_1.mean() + td_player_loss_2.mean()

        #-----------------------------------------------------------------
        # ---------- CQL Loss  ---------- ball + player
        # conservative loss


        player_action = rearrange(player_action,'(b t) a -> b t a', b=B)
        
        q_ball_1 = V_ball  + a_ball_1      # [B*T, 1, 6]
        q_ball_2 = V_ball  + a_ball_2      # [B*T, 1, 6]  
        cql_loss_ball_1 = cql_loss_logsumexp_ball(q_ball_1,ball_action,q_ball_1.device) # [B*T, 1]
        cql_loss_ball_2 = cql_loss_logsumexp_ball(q_ball_2,ball_action,q_ball_2.device) # [B*T, 1]

        # -----------add 11.25 ---------------
        ball_action_prev = torch.cat([ball_action[:, :1], ball_action[:, :-1]], dim=1) #[B,T]
        action_label = (ball_action == ball_action_prev).long() #true 表示这个动作是运球，False 表示这个位置的实际动作是传球 [B,T]
        action_label = rearrange(action_label,'b t -> (b t)').unsqueeze(-1)  #[B*T,1]
        ball_action_ = rearrange(ball_action,'b t -> (b t)').unsqueeze(-1)  #[B*T]

        #cql_weight = torch.ones_like(action_label)
        #cql_weight[action_label == 1] = 0.92  # dribble 时减弱 shot 和pass 的CQL. 
        
        
        cql_loss_ball_1 = cql_loss_ball_1 #* cql_weight
        cql_loss_ball_2 = cql_loss_ball_2 #* cql_weight
        #---------------11.25-----------------

        q_pl_1 = V_pl + nb + a_pl_1      # [B*T, 5, 19]
        q_pl_2 = V_pl  + nb + a_pl_2      # [B*T, 5, 19]
        cql_loss_pl_1 = cql_loss_logsumexp_player(q_pl_1,player_action,q_pl_1.device) #[B*T, 5]
        cql_loss_pl_2 = cql_loss_logsumexp_player(q_pl_2,player_action,q_pl_2.device) #[B*T, 5]


        if self.only_ball:
            cql_loss = ( (cql_loss_ball_1.mean() + cql_loss_ball_2.mean())/2 )
        else:
            cql_loss = ( (cql_loss_ball_1.mean() + cql_loss_ball_2.mean())/2 
                   + 5 * (cql_loss_pl_1.mean() + cql_loss_pl_2.mean())/2 )
        


        


        #---------- QSQ task  ---------- 
        qsq_gt = rearrange(qsq_gt,'(b a) t -> b a t',a = 5) #[B,5,T]
        qsq_loss = F.mse_loss(qsq_gt, qsq_pred, reduction='none') #[B,5,T]
        qsq_loss_mask = build_loss_mask_from_lengths(done, T).to(qsq_loss.device) #[B,T]
        qsq_loss_mask = qsq_loss_mask.bool().unsqueeze(1).expand(-1, 5, -1) #[B,5,T]
        qsq_loss = (qsq_loss * qsq_loss_mask.float()).mean()

        

        #---------- Action Classification task  ----------
        ball_action_prev = torch.cat([ball_action[:, :1], ball_action[:, :-1]], dim=1) #[B,T]
        action_label = (ball_action == ball_action_prev).long() #true 表示这个动作是运球，False 表示这个位置的实际动作是传球 [B,T]
        action_label = rearrange(action_label,'b t -> (b t)')  #[B*T]
        
        #action_logits  # [B*T, 2]
        valid_mask = (ball_action != 40).reshape(-1)  # [B*T]
        
        ce_loss_frame = F.cross_entropy(
            action_logits[valid_mask],   # 预测: [N_valid, 2]
            action_label[valid_mask],     # 标签: [N_valid]
            weight=torch.tensor([18.0, 1.0], device=action_logits.device),  # 给类别加权，鼓励识别传球
            reduction='none'
        )
        ce_loss = ce_loss_frame.mean()

         #---------- Print ----------

        with torch.no_grad():
            logits_flat = action_logits.reshape(-1, 2)
            labels_flat = action_label.reshape(-1)
            mask_flat = valid_mask.reshape(-1)
            preds = logits_flat.argmax(dim=-1)
            acc = (preds == labels_flat)[mask_flat].float().mean()
            acc0 = ((preds == 0) & (labels_flat == 0))[mask_flat].sum() / (labels_flat == 0)[mask_flat].sum()
            acc1 = ((preds == 1) & (labels_flat == 1))[mask_flat].sum() / (labels_flat == 1)[mask_flat].sum()
            print(f"aux acc={acc:.3f} | no-dribble acc={acc0:.3f} | dribble acc={acc1:.3f}")


        if self.global_step % 500 == 0:  # 每 500 步打印一次
            b_idx = 0  # 打印第一个 batch 的样本
            t_idx = done[b_idx] - 1  # 打印最后一帧（done帧）
            
            print(f"\n[Step {self.global_step}] QSQ Prediction Example:")
            print(f"GT   : {qsq_gt[b_idx, :, t_idx].detach().cpu().numpy()}")
            print(f"Pred : {qsq_pred[b_idx, :, t_idx].detach().cpu().numpy()}")

        warmup_steps = 1000
        if self.global_step < warmup_steps:
            total_loss = qsq_loss + ce_loss  # 仅优化 qSQ
            #total_loss = ce_loss
        else:
            total_loss = 10*td_loss + 0.1 * cql_loss + 0.1 * qsq_loss + 0.1 * ce_loss 
            #total_loss = ce_loss
        #total_loss =   0.1 * qsq_loss

        


        return total_loss,td_loss,cql_loss,qsq_loss,ce_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.Optimizer.lr,
            eps=self.cfg.Optimizer.eps,
            weight_decay=self.cfg.Optimizer.decay
        )

        # === 可配置参数 ===
        warmup_steps = 500
        total_steps = self.trainer.max_epochs * 300  # 假设每个 epoch 大约 100 step
        min_lr = 1e-5
        init_lr = self.cfg.Optimizer.lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                scale = float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                scale = 1.0 - progress
            # 🚨 clip 最小值
            return max(scale, min_lr / init_lr)
        
        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  #  一定要按 step 更新
                "frequency": 1,
                "name": "linear_warmup_decay"
            },
        }

    



if __name__ == '__main__':
    #test_valend()
    print('wx')