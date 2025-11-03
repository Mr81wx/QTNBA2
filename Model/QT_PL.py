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

# ---- 还原并拼回 25‑logit ----
def reshape_and_cat(q_ball, q_pl):
    q_ball = q_ball.squeeze(1)
    return cat_ball_player(q_ball, q_pl)                   # [B*T, 6, 25]

def cat_ball_player(q_ball, q_players):
    """
    q_ball    : [B*T, 6]           – ball‑head logits
    q_players : [B*T, P, 19]       – player‑head logits，P 可以是 1、5、甚至可变
    returns   : [B*T, 1+P, 25]
    """
    inf  = -torch.inf
    BTx  = q_ball.size(0)
    P    = q_players.size(1)       # 自动拿玩家行数
    dev  = q_ball.device
    dtype= q_players.dtype

    pad_front = torch.full((BTx, P, 6), inf, device=dev, dtype=dtype)
    players_padded = torch.cat([pad_front, q_players], dim=-1)   # [B*T, P, 25]

    pad_back = torch.full((BTx, 19), inf, device=dev, dtype=q_ball.dtype)
    ball_padded = torch.cat([q_ball, pad_back], dim=-1).unsqueeze(1)  # [B*T,1,25]

    return torch.cat([ball_padded, players_padded], dim=1)       # [B*T, 1+P, 25]


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
    
    def forward(self, batch):
        state_tokens = batch['state_tokens'] #598为无效帧补齐的
        agent_ids = batch['agent_ids']
        padding_mask = batch['padding_mask'] #[B*A,T] 
        action_tokens = batch['action_tokens'] # [B*A,T] #30为无效帧补齐的

        edge_index = batch['edge_index']  #[2,110]
        edge_attr = batch['edge_attr'] #[B,T,110,1]

        rewards = batch['rewards'] #[B,T] 实际最后一个timesteps都是shot，只有a0一个action -1e9 为无效
        done = batch['done'] # [B] 每一个回合的实际长度

        monte_carlo_return = default(batch['mc_return'], -1e4) #[B,T]

        out = self.model(state_tokens, agent_ids, padding_mask, action_tokens, edge_index, edge_attr) 
        # out =  {
        # "players": {
        #     "min": q_min_players,
        #     "avg": q_avg_players,
        #     "1":   q1_players,
        #     "2":   q2_players,
        # },
        # "ball": {
        #     "min": q_min_ball,
        #     "avg": q_avg_ball,
        #     "1":   q1_ball,
        #     "2":   q2_ball,
        # },
        # "qsq": qsq # [B,6,T,1]
        #}

        return out

    def training_step(self, batch, batch_idx):
        total_loss,td_loss,cql_loss,entropy_mean,qsq_loss,kl_loss = self.compute_loss(batch)

        self.log_dict({
            "train/total_loss": total_loss,
            "train/td_loss": td_loss,
            "train/cql_loss": cql_loss,
            "train/entropy" : entropy_mean,
            "train/qsq":qsq_loss,
            "train/kl_loss":kl_loss
            #"train/reverse_penalty": reverse_penalty,
        }, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False)
            
        return total_loss
    
    def training_step_end(self, outputs):
        print("=== Gradient Check ===")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name:<40} | grad mean: {param.grad.abs().mean():.4e} | max: {param.grad.abs().max():.4e}")
            else:
                print(f"{name:<40} | grad is None")
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
        mc_return     = default(batch['mc_return'], -1e4) #[B, T]

        B, T = rewards.shape

        # ---------- forward ----------
        out = self.model(state_tokens, agent_ids,
                        padding_mask, action_tokens,
                        edge_index, edge_attr)

        # ---------- avg-head logits ----------
        q_ball_avg   = out['ball']['avg']            # [B*T,1, 6]
        q_player_avg = out['players']['avg']         # [B*T,5, 19]
        P = 5                     # 动态拿玩家行数
        A = 6                                    # 总行数 (ball + players)

        # === 真实动作 ===
        ball_action = action_tokens[0::6, :]  # [B, T]
        player_action = rearrange(action_tokens, '(b a) t -> b t a', a=6)[..., 1:]  # [B, T, 5]
        player_action = rearrange(player_action, 'b t a -> (b t) a')  # [B*T, 5]

        # ---------- 1. Q mean ----------
        q_ball_selected = batch_ball_select_indices(q_ball_avg, action_tokens) # [B*T, 1]

        q_player_selected = batch_player_select_indices(q_player_avg, action_tokens) #[B*T, 5]

        #mask_ball = ((ball_action != 27) & (ball_action != 7)).reshape(-1) # [B*T]
        mask_ball = (ball_action != 30).reshape(-1) # [B*T]
        mask_ball = mask_ball.unsqueeze(-1)  # [B*T, 1]
        mask_player =  (player_action != 30) # [B*T,5]

        q_mean_ball = q_ball_selected[mask_ball.bool()].mean()
        q_mean_player = q_player_selected[mask_player.bool()].mean()
        
        #q_mean = 0.5 * q_mean_ball + 0.5 * q_mean_player
        self.log("val/q_mean_ball", q_mean_ball)
        self.log("val/q_mean_player", q_mean_player)

        # ---------- 2. policy accuracy ----------

      

        # === argmax 动作 ===
        q_ball_argmax = q_ball_avg.argmax(dim=-1)  # [B*T, 1]
        q_player_argmax = q_player_avg.argmax(dim=-1)  # [B*T, 5]

        # reshape ball_action: [B, T] → [B*T, 1]
        ball_action_flat = ball_action.reshape(-1, 1)          # [B*T, 1]

        # === accuracy ===
        acc_ball = (q_ball_argmax == ball_action_flat)[mask_ball].float().mean()
        acc_player = (q_player_argmax == player_action)[mask_player].float().mean()

        self.log("val/acc_ball", acc_ball)
        self.log("val/acc_player", acc_player)
        
        # ---------- 3. L1( Q_p5 , MC ) ----------
        q_p5_selected = q_player_selected[...,-1] #[B,T]
        q_p5_selected = rearrange(q_p5_selected, '(b t) -> b t', b=B)  # [B, T]
        
        mc_return = mc_return.squeeze(-1)  # [B,T]
        l1 = F.l1_loss(q_p5_selected, mc_return, reduction='none')  # [B,T]
        
        mask_p5 = mask_player[..., -1]  # [B*T]
        mask_p5 = rearrange(mask_p5, '(b t) -> b t', b=B)  # [B, T]
        l1 = l1[mask_p5.bool()].mean()

        self.log("val/l1_q_vs_mc", l1, prog_bar=True)



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

        rewards  = batch['rewards']               # [B, T]
        done     = batch['done']                  # [B]
        mc_ret   = batch['mc_return']             # [B, T]
        edge_idx = batch['edge_index']            # [2,132]
        edge_attr= batch['edge_attr']             # [B,T,132,1]

        qsq_gt = batch['qsq']                     # [B*5,T]

        B, T = rewards.shape
        Y = self.discount_factor_gamma

        # #state mask， 对于在state中有球员在后场的情况都不去计算该球员的移动loss，{TODO_ 同时对于球的行为7的不去计算这一帧与球相关的loss}
        # state_x = state_tokens[...,0] #[B*A,T]
        # state_x = rearrange(state_x,'(b a) t -> b t a',b=B)
        # state_x = state_x[:,:,:6] # [B,T,5]
        # #invalid_state_mask = (state_x > 53).float()  # [B, T， 5]，
        # invalid_state_mask = (state_x > 53).any(dim=-1)  # [B, T] 对任意state中有进攻球员在后场，都直接不算其loss
        
        # 构造每个样本 done-1 的索引
        # batch_idx = torch.arange(B, device=done.device)  # [B]
        # last_valid_idx = done - 1                        # [B]

        # 在 invalid_state_mask 中将 done-1 的帧置为 False,确保投篮帧的loss一定考虑
        # invalid_state_mask = invalid_state_mask.clone()
        # invalid_state_mask = invalid_state_mask.scatter(dim=1, index=last_valid_idx.unsqueeze(1), value=False)


        # ---------- 前向 ----------
        # self.model 输出嵌套字典，两套 head
        out_pred = self.model(state_tokens, agent_ids,
                            padding_mask, action_tokens,
                            edge_idx, edge_attr)

        # players: [ B*T,5, 19]    ball: [ B*T,1, 8]
        q_pl_1   = out_pred['players']['1']  #[ B*T,5, 19] 
        q_pl_2   = out_pred['players']['2']
        q_pl_avg = out_pred['players']['avg']
        
        q_ball_1 = out_pred['ball']['1'] #[ B*T,1, 6]
        q_ball_2 = out_pred['ball']['2']
        q_ball_avg = out_pred['ball']['avg']

        qsq_pred = out_pred['qsq'].squeeze(-1) # [B,5,T]
        #print("qsq_pred:", qsq_pred.shape)

        # ---------- select 本步 Q ----------

        q_pred_1_ball = batch_ball_select_indices(q_ball_1, action_tokens) # [B*T, 1]
        q_pred_1_ball = rearrange(q_pred_1_ball, '(b t) n -> b t n', b=B)

        q_pred_2_ball = batch_ball_select_indices(q_ball_2, action_tokens)
        q_pred_2_ball = rearrange(q_pred_2_ball, '(b t) n -> b t n', b=B)

        q_pred_1_player = batch_player_select_indices( q_pl_1,action_tokens) #[B*T, 5]
        q_pred_1_player = rearrange(q_pred_1_player, '(b t) n -> b t n', b=B)

        q_pred_2_player = batch_player_select_indices( q_pl_2,action_tokens) #[B*T, 5]
        q_pred_2_player = rearrange(q_pred_2_player, '(b t) n -> b t n', b=B) 

        q_pred_1 = torch.cat([q_pred_1_ball, q_pred_1_player], dim=-1)
        q_pred_2 = torch.cat([q_pred_2_ball, q_pred_2_player], dim=-1)
        
       
        # ---------- target 网络 ----------
        ball_action = action_tokens[0::6,:] #[B,T] action_tokens [B*6,T]
        valid_mask_ball =  (ball_action != 30) #[B,T] 30为无效帧补齐的
        valid_mask_ball = valid_mask_ball.reshape(-1, 1, 1).expand(-1, 1, 6)  #[B*T,1,6]

        #ball_action7 = (ball_action == 7) #[B,T]
        #mask = rearrange(ball_action7, 'b t -> (b t) 1 1')  #[B*T,1,8]
        #mask = mask.expand(-1, 1, 8)                        # [B*T, 1, 8]
        #action7_mask = mask.float()
        # 把 bin=7 的位置改回 0
        #action7_mask = action7_mask.scatter(dim=-1, index=torch.tensor([7], device=action7_mask.device).unsqueeze(0).unsqueeze(0), value=0.0) # True for mask  

        #mask_ = (~ball_action7).reshape(-1, 1, 1).expand(-1, 1, 8).float()
        #mask_ = mask_.scatter(dim=-1, index=torch.arange(7, device=mask_.device).unsqueeze(0).unsqueeze(0), value=0.0) # True for mask   把正常frame中的action设置为-1e9，这样max的不会选择这个action

        #action7_mask = action7_mask + mask_

        player_action = rearrange(action_tokens,'(b a) t -> b a t',a = 6)[:,1:,:]
        player_action = rearrange(player_action, 'b a t -> (b t) a') #[B*T,5]
        valid_mask_player = (player_action != 30) #[B*T,5]
        valid_mask_player = valid_mask_player.unsqueeze(-1).expand(-1, -1, 19) #[B*T,5,19]

        with torch.no_grad():
            out_target = self.ema_model(state_tokens, agent_ids,
                                        padding_mask, action_tokens,
                                        edge_idx, edge_attr)
            q_pl_target   = out_target['players']['min']   # 取 avg 头 or min [ B*T,5, 19] 
            q_ball_target = out_target['ball']['min']  #[ B*T,1, 6] 

            q_pl_target = q_pl_target * valid_mask_player.float() #把无效帧的q值设置为0
            q_ball_target = q_ball_target * valid_mask_ball.float() #把无效帧的q值设置为0
                        
            q_ball_target_max = q_ball_target.max(dim=-1).values          # [B*T, 1]
            q_pl_target_max = q_pl_target.max(dim=-1).values  #[B*T,5]

            q_target = torch.cat([q_ball_target_max, q_pl_target_max], dim=-1) #[B*T, 6]
            q_target = rearrange(q_target, '(b t) n -> b t n', b=B) # [B, T, 6]

            mc_ret = mc_ret.squeeze(-1)  # [B,T]
            mc_ret   = repeat(default(mc_ret, -10), 'b t -> (b t) n', n=6) # [B*T, 6]
            #q_target = q_target.clamp(min=mc_ret.reshape(B, T, 6))

        # ---------- 分离predict [ball,p1,p2,p3,p4] [p5]   ----------
        # ---------- 分离target  [p1,p2,p3,p4,p5]   [ball] ----------
        #            对应计算loss ball_pred <-> p1_target .... p5_pred <-> r + y * ball_target

        # q_pred_rest = index 0-5 ,q_pred_last = index 6
        q_pred_rest_1, q_pred_last_1 = q_pred_1[..., :-1], q_pred_1[..., -1] #[B，T, 5]，[B,T]
        q_pred_rest_2, q_pred_last_2 = q_pred_2[..., :-1], q_pred_2[..., -1] #[B，T, 5]，[B,T]
        q_target_first, q_target_rest = q_target[..., 0], q_target[..., 1:] #[B, T]，[B*T, 5]

        # q_pred_rest 和 q_target_rest 求td loss， q_pred_last 和 q_target_first 求td loss
        # smooth‑l1 for ball + player 1-4 
        losses_rest_1 = F.smooth_l1_loss(q_pred_rest_1, q_target_rest,
                                        reduction='none') #[B，T, 5]
        losses_rest_2 = F.smooth_l1_loss(q_pred_rest_2, q_target_rest,
                                        reduction='none') #[B，T, 5]

        
        #loss_1_mask = build_loss_mask_from_lengths(done-1, T).to(losses_rest_1.device) #[B,T] 把时间维度 T-1 之后都设置0，因为投篮之后球员的动作不影响reward
        loss_1_mask = build_loss_mask_from_lengths(done-1, T).to(losses_rest_1.device) #[B,T] 先尝试不管投篮之后的帧

        final_loss_1_mask = loss_1_mask.bool() 
        final_loss_1_mask = final_loss_1_mask.unsqueeze(-1).expand(-1, -1, 5)

        loss_1_1 = losses_rest_1 * final_loss_1_mask.float()
        loss_1_2 = losses_rest_2 * final_loss_1_mask.float()  #[B,T,5]

        # last‑step ball TD
        q_target_ball = rewards[...,:-1] + Y * q_target_first[...,1:] #[B,T-1]

        # add final reward to q_target_first[B,done] 修改终结帧的q_target 直接等于reward
        #batch_idx = torch.arange(B)
        #final_rewards = rewards[batch_idx, done - 1]

        #valid_idx = done >= 2  # 防止 done=1 时索引 < 0

        # 使用非原地操作替换原地赋值
        #q_target_ball = q_target_ball.clone()
        #batch_indices = batch_idx[valid_idx]
        #done_indices = done[valid_idx] - 2
        #final_rewards_valid = final_rewards[valid_idx]
        
        # 使用scatter进行非原地赋值
        #indices = torch.stack([batch_indices, done_indices], dim=1)
        #q_target_ball = q_target_ball.scatter(dim=0, index=indices, src=final_rewards_valid.unsqueeze(1))

        #q_pred_last_1 = q_pred_last_1[...,:-1]
        #q_pred_last_2 = q_pred_last_2[...,:-1]

        #loss_2_1 = F.smooth_l1_loss(q_pred_last_1, q_target_ball,reduction='none') #[B,T-1]
        #loss_2_2 = F.smooth_l1_loss(q_pred_last_2, q_target_ball,reduction='none') #[B,T-1]

        #只看球
        q_pred_first_1 = q_pred_1[...,0] #[B,T]
        q_pred_first_2 = q_pred_2[...,0] #[B,T]
        q_pred_first_1 = q_pred_first_1[...,:-1] #[B,T-1]
        q_pred_first_2 = q_pred_first_2[...,:-1] #[B,T-1]

    
        loss_2_1 = F.smooth_l1_loss(q_pred_first_1, q_target_ball,reduction='none') #[B,T-1]
        loss_2_2 = F.smooth_l1_loss(q_pred_first_2, q_target_ball,reduction='none') #[B,T-1]
        
        loss_2_mask = build_loss_mask_from_lengths(done-1, T-1).to(loss_2_1.device) # [B,T-1] 

        #invalid_state_mask_1 = invalid_state_mask[:,:-1]
        #invalid_state_mask_2 = invalid_state_mask[:,1:]
        #invalid_td_mask = invalid_state_mask_1 | invalid_state_mask_2 #true for padding

        final_loss_2_mask = loss_2_mask.bool()# & (~invalid_td_mask.bool())

        loss_2_1 = loss_2_1 * final_loss_2_mask.float()#[B,T-1]
        loss_2_2 = loss_2_2 * final_loss_2_mask.float()#[B,T-1]


        #shot loss 直接监督最后一帧的q值等于reward #[B，T, 6]

        batch_indices = torch.arange(B, device=done.device)
        time_indices = done - 1  # shape: [B]
        q_pred_shot_1 = q_pred_1[batch_indices, time_indices, 0]  # [B]
        q_pred_shot_2 = q_pred_2[batch_indices, time_indices, 0]  # [B]
        final_rewards = rewards[batch_indices, time_indices]     # [B]

        #mu = torch.tensor(0.8191, device=final_rewards.device)
        #alpha = torch.tensor(2.5, device=final_rewards.device)

        #final_rewards = mu + alpha * (final_rewards - mu)


        loss_3_1 = F.smooth_l1_loss(q_pred_shot_1, final_rewards, reduction='none') #[B]
        loss_3_2 = F.smooth_l1_loss(q_pred_shot_2, final_rewards, reduction='none') #[B]


        #------------球员原地踏步的action 降低权重 --------- action_token = 6
        # action_player_ = rearrange(action_tokens,'(b a) t -> b t a',b = B) #[B,T,6]
        # bin_8_mask = (action_player_ == 8).float() #[B,T,6]

        # bin_8_mask_1 = bin_8_mask[...,:-1] #ball,p1-p4的mask，针对loss_1_ #[B,T,5]
        # loss_weight_1 = 1.0 - 0.5 * bin_8_mask_1  # 把 bin=8 的 loss 权重调成 0.5，其余为 1.0
        # loss_1_1 = loss_1_1 * loss_weight_1  # [B, T, 5] × [B, T, 5]
        # loss_1_2 = loss_1_2 * loss_weight_1

        # bin_8_mask_2 = bin_8_mask[:,:-1,-1] #p5 [B,T-1]
        # loss_weight_2 = 1.0 - 0.5 * bin_8_mask_2  # 把 bin=8 的 loss 权重调成 0.5，其余为 1.0
        # loss_2_1 = loss_2_1 * loss_weight_2  # [B, T, 5] × [B, T, 5]
        # loss_2_2 = loss_2_2 * loss_weight_2


        #summmary td_loss

        td_loss_1 = loss_1_1.mean() + loss_1_2.mean()
        td_loss_2 = loss_2_1.mean() + loss_2_2.mean()
        td_loss_3 = loss_3_1.mean() + loss_3_2.mean()
        
        #td_loss = (td_loss_1 + td_loss_2 + td_loss_3)/2

        td_loss = (
    (loss_2_1.sum() + loss_2_2.sum() + loss_3_1.sum() + loss_3_2.sum())
    / (loss_2_1.numel() + loss_2_2.numel() + loss_3_1.numel() + loss_3_2.numel())
)
        
        #-----------------------------------------------------------------

        # conservative loss

        # pred player ： q_pl_1 q_pl_2 [ B*T,5, 19] 
        # pred ball ： q_bal_1 q_ball_2 [B*T,1,8]

        # player_action: [B*T，5]
        # ball_action: [B,T]
    

        # padding_mask : player + ball 重构一个？

        cql_loss_player_1 = cql_loss_hard_player(q_pl_1,player_action,self.min_reward,q_pl_1.device) #[B*T, 5, 19]
        cql_loss_player_2 = cql_loss_hard_player(q_pl_2,player_action,self.min_reward,q_pl_1.device) #[B*T, 5, 19]

        

        #cql_loss_ball_1 = cql_loss_hard_ball(q_ball_1,ball_action,self.min_reward,q_pl_1.device) # [B*T, 1, 6]
        #cql_loss_ball_2 = cql_loss_hard_ball(q_ball_2,ball_action,self.min_reward,q_pl_1.device) # [B*T, 1, 6]
        cql_loss_ball_1 = cql_loss_logsumexp_ball(q_ball_1,ball_action,q_ball_1.device) # [B*T, 1]
        cql_loss_ball_2 = cql_loss_logsumexp_ball(q_ball_2,ball_action,q_ball_2.device) # [B*T, 1]

        #state mask，剔除游球员和球在后场时的帧
        #mask_cql_player = rearrange(invalid_state_mask, 'b t -> (b t) 1 1')  # [B*T,1,1]
        # mask_cql_player = build_loss_mask_from_lengths(done, T).to(cql_loss_player_1.device) #[B,T] 
        # mask_cql_player = rearrange(mask_cql_player, 'b t -> (b t) 1 1')  # [B*T,1,1]
        # cql_loss_player_1 = cql_loss_player_1 * (~mask_cql_player.expand_as(cql_loss_player_1)).float()
        # cql_loss_player_2 = cql_loss_player_2 * (~mask_cql_player.expand_as(cql_loss_player_2)).float()

        # cql_loss_ball_1 = cql_loss_ball_1 * (~mask_cql_player.expand_as(cql_loss_ball_1)).float()
        # cql_loss_ball_2 = cql_loss_ball_2 * (~mask_cql_player.expand_as(cql_loss_ball_2)).float()
        
        # print("Valid CQL player elements:", (cql_loss_player_1 != 0).sum().item())
        # print("Valid CQL ball elements:", (cql_loss_ball_1 != 0).sum().item())

        cql_loss_1 =  cql_loss_player_1.mean() + cql_loss_ball_1.mean()
                      
        cql_loss_2 =  cql_loss_player_2.mean() + cql_loss_ball_2.mean()

        cql_loss = (cql_loss_1 + cql_loss_2) / 2
        #cql_loss = cql_loss_2


        #-----------------------------------------------------------------

        # conservative loss     
        # entropy_player_1 = entropy_reg_player(q_pl_1, player_action)
        # entropy_player_2 = entropy_reg_player(q_pl_2, player_action)

        entropy_ball_1 = entropy_reg_ball(q_ball_1, ball_action)
        entropy_ball_2 = entropy_reg_ball(q_ball_2, ball_action)

        #entropy_mean = entropy_player_1 + entropy_player_2 + entropy_ball_1 + entropy_ball_2
        entropy_mean =  entropy_ball_1 + entropy_ball_2


        #qsq predict loss 
        qsq_gt = rearrange(qsq_gt,'(b a) t -> b a t',a = 5) #[B,5,T]
        qsq_loss = F.mse_loss(qsq_gt, qsq_pred, reduction='none') #[B,5,T]
        qsq_loss_mask = build_loss_mask_from_lengths(done, T).to(qsq_loss.device) #[B,T]
        qsq_loss_mask = qsq_loss_mask.bool().unsqueeze(1).expand(-1, 5, -1) #[B,5,T]
        qsq_loss = (qsq_loss * qsq_loss_mask.float()).mean()

        #KL loss
        kl_loss_q1 = compute_kl_regularizer(q_ball_1, ball_action, alpha_kl=0.05)
        kl_loss_q2 = compute_kl_regularizer(q_ball_2, ball_action, alpha_kl=0.05)

        kl_loss = (kl_loss_q1 + kl_loss_q2) /2

        if self.global_step % 50 == 0:  # 每 500 步打印一次
            b_idx = 0  # 打印第一个 batch 的样本
            t_idx = done[b_idx] - 1  # 打印最后一帧（done帧）
            
            print(f"\n[Step {self.global_step}] QSQ Prediction Example:")
            print(f"GT   : {qsq_gt[b_idx, :, t_idx].detach().cpu().numpy()}")
            print(f"Pred : {qsq_pred[b_idx, :, t_idx].detach().cpu().numpy()}")

        # total_loss = (self.cfg.td_loss_coef * td_loss +
        #             self.cfg.cql_loss_coef * cql_loss -
        #             0 * entropy_mean + 0.1 * qsq_loss)
        total_loss = td_loss +  cql_loss + 0.5 * qsq_loss - 0.0 * entropy_mean + kl_loss
        #total_loss =   0.1 * qsq_loss

        if self.global_step % 50 == 0:
            with torch.no_grad():
                # ---- Ball Q 值直方图 ----
                q_ball_pred = q_ball_1.detach()  # [B*T, 1, 8]
                valid_q_ball = q_ball_pred[valid_mask_ball]  # [N, 8]
                self.logger.experiment.add_histogram(
                    tag="q_values/ball",
                    values=valid_q_ball,
                    global_step=self.global_step
                )

                # ---- Player Q 值直方图 ----
                q_player_pred = q_pl_1.detach()  # [B*T, 5, 19]
                for pid in range(5):  # agent 1 ~ 5
                    mask_pid = valid_mask_player[:, pid]  # [B*T]
                    q_valid_pid = q_player_pred[:, pid, :][mask_pid]  # [N, 19]
                    self.logger.experiment.add_histogram(
                        tag=f"q_values/player{pid+1}",
                        values=q_valid_pid,
                        global_step=self.global_step
                    )

                # ---- Ball argmax bin ----
                q_ball_argmax = q_ball_pred.argmax(dim=-1).reshape(-1)  # [B*T]
                q_ball_argmax_valid = q_ball_argmax[valid_mask_ball[..., 0].squeeze(-1)]  # [N]

                num_bins = q_ball_pred.shape[-1]  # 比如8
                bin_counts = torch.bincount(q_ball_argmax_valid, minlength=num_bins)

                # 打印统计结果
                print(f"Ball action bin counts: {bin_counts.tolist()}")
                self.logger.experiment.add_histogram(
                    tag="actions/ball_argmax_bin",
                    values=q_ball_argmax_valid,
                    global_step=self.global_step
                )

                # ---- Player argmax bin ----
                q_player_argmax = q_player_pred.argmax(dim=-1)  # [B*T, 5]
                for pid in range(5):
                    argmax_valid = q_player_argmax[:, pid][valid_mask_player[:, pid, 0]]
                    self.logger.experiment.add_histogram(
                        tag=f"actions/player{pid+1}_argmax_bin",
                        values=argmax_valid,
                        global_step=self.global_step
                    )


        return total_loss,td_loss,cql_loss,entropy_mean,qsq_loss,kl_loss


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