HanoiWorld

HanoiWorld is a driving simulation benchmark built on top of DreamerV3 (PyTorch), designed to evaluate world models and representation learning methods (JEPA, RSSM, VQ-VAE) in structured urban driving scenarios.

This repository integrates:

DreamerV3-Torch for world model learning and planning

JEPA-based encoders for representation learning

Custom driving environments inspired by real-world traffic scenarios

🌍 Environments

HanoiWorld consists of three driving environments, each targeting different planning and perception challenges:

envs/
├── highway/       # High-speed lane keeping and collision avoidance
├── roundabout/    # Multi-agent interaction and yielding behavior
└── merge/         # Lane merging and gap acceptance

Environment Overview
Environment	Key Challenges
Highway	Long-horizon stability, collision avoidance
Roundabout	Multi-agent reasoning, right-of-way
Merge	Decision making under uncertainty

Each environment provides:

Image observations

Structured metadata (collision, off-road, goal reached)

Continuous control actions

🧠 Methods

HanoiWorld supports the following world model components:

RSSM (Recurrent State Space Model) — temporal latent dynamics

JEPA (Joint Embedding Predictive Architecture) — self-supervised visual representation learning

VQ-VAE (optional baseline) — discrete latent representations

💡 In this setup, models are typically pretrained and then plugged into Dreamer for evaluation, not retrained end-to-end.

🚀 Installation
Requirements

Python 3.11

PyTorch (CUDA recommended)

Gymnasium-compatible environments

Install dependencies:

pip install -r requirements.txt

🏋️ Training
DreamerV3 (baseline)

Example: train DreamerV3 on a HanoiWorld environment

python dreamer.py \
  --configs hanoiworld \
  --task highway \
  --logdir ./logdir/highway


Monitor training:

tensorboard --logdir ./logdir


⚠️ In most experiments, JEPA models are pretrained and reused for evaluation.

📊 Evaluation

Evaluate a pretrained JEPA + RSSM model inside Dreamer:

Evaluation metrics include:

Collision rate

Off-road rate

Success rate

Mean lateral deviation

minADE (trajectory error)

Driving score

📈 Benchmarks

HanoiWorld is inspired by and complements existing benchmarks:

Benchmark	Observation	Action	Budget
DMC Vision	Image	Continuous	1M
Atari 100k	Image	Discrete	400K
Crafter	Image	Discrete	1M
HanoiWorld	Image + State	Continuous	Variable
🧩 DreamerV3 Integration

This project is based on the PyTorch implementation of DreamerV3:

DreamerV3: Mastering Diverse Domains through World Models
https://arxiv.org/abs/2301.04104v1

Original repository:
https://github.com/Min34r/dreamerv3-torch

Dreamer components (RSSM, actor, critic, imagination) remain largely unchanged.
Custom logic is introduced via:

Encoder replacement (JEPA)

Environment wrappers

Evaluation scripts

🙏 Acknowledgments

This project builds upon:

DreamerV3 (JAX): https://github.com/danijar/dreamerv3

DreamerV2 (TF): https://github.com/danijar/dreamerv2

DreamerV2 (PyTorch): https://github.com/jsikyoon/dreamer-torch

DrQ-v2: https://github.com/facebookresearch/drqv2

JEPA: https://arxiv.org/abs/2301.08243