environment //
  PyTorch  2.1.2 //
  Python  3.10(ubuntu22.04) //
  CUDA  11.8 //
  pytorch-lightning 2.2.1 //


data format #每一个.pt文件代表一个回合 //
{
    "state_tokens": [T, 12, 1],     # ball + players + rim, 
    "agent_ids": [11]               # ball = 0, rim = 500,
    "padding_mask"  [12, T]         # padding for encoder temporal attention,
    "action_tokens": [T, 6],        # ball & player discrete bins,
    "rewards": [T],                 # reward_t = qsq_t+1 - qsq_t 
    "done": [T],                    # episode length
    "edge_index": [2, E],           # graph edges
    "edge_attr": [E,T,D],            # edge attr [,132，3] 全连接图，
    "qsq": [T,5]
}

config 
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
  

python train_dueling_pointerdecoder.py \
    model=QTransformerSoftPointer \
    data.path=/path/to/your/data \
    trainer.max_epochs=30 (30mins 1 epoch)


