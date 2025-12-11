# [Distill] No distilled weights found at bev_mobilenet_dino_init.pt, training JEPA from random init.
# ======== JEPA WORLD MODEL CONFIG ========
JEPA_WorldModel(
  (jepa1): PrimitiveLayer(
    (context_encoder): BEVJEPAEncoder2D(
      (s1): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
          (dw_bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
          (dw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s2): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=8, bias=False)
          (dw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          (dw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s3): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
          (dw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (dw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s4): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (dw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (dw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
    )
    (target_encoder): BEVJEPAEncoder2D(
      (s1): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
          (dw_bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
          (dw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s2): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=8, bias=False)
          (dw_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          (dw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s3): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
          (dw_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (dw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
      (s4): Sequential(
        (0): MobileNetBlock(
          (depthwise): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (dw_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
        (1): MobileNetBlock(
          (depthwise): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (dw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pointwise): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pw_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): GELU(approximate='none')
        )
      )
    )
    (predictor): SpatialPredictorCNN(
      (conv): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): GELU(approximate='none')
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): GELU(approximate='none')
        (4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (jepa2): Tier2Module(
    (trajectory_tok): TrajectoryTokenizerFSQ(
      (encoder): GatedCNNEncoder(
        (conv_stack): Sequential(
          (0): GatedConvBlock(
            (conv_feat): Conv1d(6, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_gate): Conv1d(6, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): GatedConvBlock(
            (conv_feat): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_gate): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): GatedConvBlock(
            (conv_feat): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_gate): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (pool): AdaptiveAvgPool1d(output_size=1)
        (proj): Linear(in_features=64, out_features=128, bias=True)
      )
      (ln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (bottleneck): Linear(in_features=128, out_features=6, bias=True)
      (quantizer): FSQ()
      (decoder): GatedCNNDecoder(
        (fc): Linear(in_features=6, out_features=512, bias=True)
        (deconv_stack): Sequential(
          (0): GatedConvBlock(
            (conv_feat): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_gate): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): GatedConvBlock(
            (conv_feat): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_gate): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (out_conv): Conv1d(64, 6, kernel_size=(1,), stride=(1,))
      )
    )
    (token_emb): TokenEmbedding(
      (embed): Embedding(1024, 256)
    )
    (traj_enc): TrajEncoder(
      (conv): Conv1d(6, 64, kernel_size=(3,), stride=(1,), padding=(1,))
      (gelu): GELU(approximate='none')
      (fc): Linear(in_features=64, out_features=128, bias=True)
    )
    (gcn): GCN(
      (conv1): GCNConv(13, 128)
      (conv2): GCNConv(128, 128)
    )
    (joint_synthesizer): JointSynthesizerFusion(
      (score_mlp): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=1, bias=True)
      )
      (value_proj): Linear(in_features=128, out_features=128, bias=True)
    )
  )
  (jepa3_inv): JEPA_Tier3_InverseAffordance(
    (kin): DeterministicKinematicBicycle(
      (expand): Sequential(
        (0): Linear(in_features=4, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=64, bias=True)
      )
    )
    (temporal_enc): TemporalActionEncoder(
      (layers): ModuleList(
        (0): GCNNBlock(
          (conv): Conv1D_Gated(
            (conv_a): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_b): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (to_token): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (beta_proj): Linear(in_features=128, out_features=128, bias=True)
          (global_lin): Linear(in_features=128, out_features=128, bias=True)
          (gamma_summary): Linear(in_features=128, out_features=128, bias=True)
        )
        (1-3): 3 x GCNNBlock(
          (conv): Conv1D_Gated(
            (conv_a): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_b): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (to_token): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (beta_proj): Linear(in_features=128, out_features=128, bias=True)
          (global_lin): Linear(in_features=128, out_features=128, bias=True)
          (gamma_summary): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
    (film_beta_proj): Linear(in_features=128, out_features=128, bias=True)
    (film_gamma_proj): Linear(in_features=128, out_features=128, bias=True)
    (spatial_encoder): SpatialEncoderFiLM(
      (conv_in): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (res_blocks): ModuleList(
        (0-3): 4 x Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (film_beta_proj): ModuleList(
        (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
      )
      (film_gamma_proj): ModuleList(
        (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
      )
      (project): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): GELU(approximate='none')
    )
    (spatial_proj): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (action_proj): Linear(in_features=128, out_features=128, bias=True)
    (z_proj): Sequential(
      (0): Linear(in_features=256, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=128, bias=True)
    )
    (pred_sy): PredictorMLP(
      (net): Sequential(
        (0): Linear(in_features=384, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (pred_tg_1): PredictorMLP(
      (net): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (pred_tg_2): PredictorMLP(
      (net): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (ema_target): TemporalActionEncoder(
      (layers): ModuleList(
        (0): GCNNBlock(
          (conv): Conv1D_Gated(
            (conv_a): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_b): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (to_token): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (beta_proj): Linear(in_features=128, out_features=128, bias=True)
          (global_lin): Linear(in_features=128, out_features=128, bias=True)
          (gamma_summary): Linear(in_features=128, out_features=128, bias=True)
        )
        (1-3): 3 x GCNNBlock(
          (conv): Conv1D_Gated(
            (conv_a): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
            (conv_b): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
          )
          (to_token): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (beta_proj): Linear(in_features=128, out_features=128, bias=True)
          (global_lin): Linear(in_features=128, out_features=128, bias=True)
          (gamma_summary): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
  )
  (jepa3_glob): JEPA_Tier3_GlobalEncoding(
    (cube_online): CubeMLP(
      (mix_L): Sequential(
        (0): Linear(in_features=6, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=6, bias=True)
      )
      (mix_M): Sequential(
        (0): Linear(in_features=4, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=4, bias=True)
      )
      (mix_D): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=128, bias=True)
      )
      (out): Linear(in_features=128, out_features=128, bias=True)
    )
    (cube_target): CubeMLP(
      (mix_L): Sequential(
        (0): Linear(in_features=6, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=6, bias=True)
      )
      (mix_M): Sequential(
        (0): Linear(in_features=4, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=4, bias=True)
      )
      (mix_D): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=128, out_features=128, bias=True)
      )
      (out): Linear(in_features=128, out_features=128, bias=True)
    )
    (global_gcn): GCN_PYG(
      (conv1): GCNConv(32, 128)
      (conv2): GCNConv(128, 128)
    )
    (pred_from_ctx): PredictorMLP(
      (net): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
      )
    )
    (s_c_proj): Linear(in_features=256, out_features=128, bias=True)
  )
)

# THE OUTCOME OF THE JEPA_TIER-3-FINAL LAYER: B x 128
========================================

✅ Parameter summary:
Total params     : 4,081,651
Trainable params : 3,346,558