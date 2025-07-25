# DGAIL: Diffusion-reward Imitation Learning for Multi-hop Task Offloading in Collaborative Edge Computing

This repository contains the simulation code for our paper  
**"Multi-hop Task Offloading in Collaborative Edge Computing: A Diffusion-reward Imitation Learning Approach"**, submitted to *IEEE Transactions on Mobile Computing* (Under review).

We propose **DGAIL**, a novel diffusion-based generative adversarial imitation learning framework designed for decentralized multi-hop task offloading in collaborative edge networks. DGAIL introduces a conditional diffusion discriminator and a structure-aware reward mechanism to improve policy learning stability and task success rates under dynamic and partially observable environments.

This repo provides the full implementation of DGAIL and representative baselines (e.g., PPO, GAIL, D2SAC, ILETS), including training pipelines, evaluation scripts, and reproducible experiments in discrete multi-agent offloading environments.



## Method

We propose **DGAIL**, a novel diffusion-based generative adversarial imitation learning framework that addresses the multi-hop task offloading problem in collaborative edge computing networks. Our approach introduces:

![](.\assets\method_01.png)

- **Conditional Diffusion Discriminator**: Leverages historical action probabilities instead of Gaussian noise for better decision-making
- **Structure-aware Reward Mechanism**: Improves policy learning stability in dynamic environments
- **Multi-agent Coordination**: Enables efficient collaboration among distributed edge nodes

The framework supports multiple state-of-the-art algorithms including PPO, SAC, GAIL, DGAIL, ILETS, and D2SAC for comprehensive performance comparison.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BMDACMER/marl.git
cd marl

# Install dependencies
pip install torch numpy loguru tensorboard matplotlib networkx
```

### Running PPO Algorithm

To quickly run the PPO algorithm with default settings:

```bash
python main.py --name=ppo --task_size_max=6000 --seed=100 --test_interval=100 --test_nepisode=5 --folder=seed_episode_paper_v0.1 --log_tag="task_size_max_6000_seed_100"
```

To run another algorithm, simply replace `ppo` with the algorithm you want to run.

## Project Structure

```
marl/
├── main.py                 # Main entry point
├── rl/                     # RL algorithm implementations
│   └── policy_gradient_rl/ # Policy gradient algorithms (PPO, SAC, GAIL, DGAIL, ILETS)
├── envs/                   # Environment implementations
│   └── edge_computing/     # Edge computing environment
├── optimal/                # Optimal search algorithms
├── runners/                # Training/testing runners
├── buffer/                 # Experience replay buffers
├── utils/                  # Utility functions
└── assets/                 # Method overview and figures
```

The detailed documents will be improved after the article is received.

## Contact

For questions and support, please contact: seguohao@mail.scut.edu.cn. Thank you for your interest in our work.
