PyTorch             2.1.2   
Python              3.10 (Ubuntu 22.04)  
CUDA                11.8  
PyTorch Lightning   2.2.1  


data format #每一个.pt文件代表一个回合   
```
{
    "state_tokens": [T, 12, 1],     
        # 12 agents: ball + 5 offense + 5 defense + rim
        # hexagon index for position

    "agent_ids": [12],                 
        # agent type encoding:
        #   ball = 0, players = 1~10, rim = 500

    "padding_mask": [12, T],           
        # temporal padding mask for attention

    "action_tokens": [T, 6],           
        # ball & player discrete bins
        # ball: 0~5 ; players: 6~24 ; 30 = padding

    "rewards": [T],                    
        # shaped reward:
        #   reward_t = qsq[t+1] - qsq[t]

    "done": [T],                       
        # episode termination marker

    "edge_index": [2, E],              
        # fully-connected graph edges
        # typical E = 132

    "edge_attr": [T, E, D],            
        # edge features:
        #   D = 3 → (dx, dy, distance)
        #   shape example: [T,132,3]

    "qsq": [T, 5]                      
        # 辅助任务标签
}
```


config 
```
{  
    # setting
    device_ids: 0
    max_epochs: 30
    batchsize: 96
    flag: 0
    check_point_name: 1116_ShareV_6actions
    only_ball: False
    
    # === Optimizer Settings ===
    Optimizer:
      lr: 1e-3
      eps: 1e-5
      decay: 0.0001
    
    # === Reinforcement Learning Settings ===
    discount_factor_gamma: 0.99
    min_reward: -3.0
    td_loss_coef: 1.0
    cql_loss_coef: 1.0
    
    soft_alpha: 0.1
    # === Model Architecture ===
    qtransformer:
      device: cuda:0
      in_feat_dim: 1
      time_steps: 80
      embed_dim: 128
      num_bins: 30 # 0-5 ball, 6-24 player 30 padding
      num_bins_player: 19
      num_bins_ball: 6
      num_actions: 6         # a0 ~ a5
      max_tokens: 32         # for positional embedding
      decoder_layers: 3
      n_head: 8
      encoder_n_head: 4
      dropout: 0.1
    
    # === EMA Settings ===
    ema:
      beta: 0.99
      update_after_step: 10
      update_every: 5
}
```  
ball action: 0 shot, 1-5 pass to 1-5(Original Order)
player action: 0 stay, 10-17, 19-26, 28-35 (invalid action id: 1-9,18,27)
<img width="515" height="493" alt="fdeb4d4b54e8d5bbb1dd8211e8a0e1a6" src="https://github.com/user-attachments/assets/072217ea-4daa-4768-a251-444345d2f341" />

