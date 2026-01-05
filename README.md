<div align="center">

# MASPO: Unifying Gradient Utility, Probability Mass, and Signal Asymmetry for Robust and Sample-Efficient LLM Reasoning

**Mass-Adaptive Soft Policy Optimization (MASPO)** - Official Implementation

</div>

## 📖 Abstract

Current Reinforcement Learning with Verifiable Rewards (RLVR) paradigms, such as GRPO, rely on rigid, uniform, and symmetric trust region mechanisms that are fundamentally misaligned with the complex optimization dynamics of Large Language Models (LLMs). In this paper, we identify three critical disconnects in existing methods: (1) **inefficient gradient utilization** caused by the binary cutoff of hard clipping, (2) **probability mass insensitivity** arising from uniform ratio constraints that ignore the token distribution, and (3) **asymmetric signal reliability** stemming from the disparate credit assignment ambiguity between positive and negative samples. To bridge these gaps, we propose **Mass-Adaptive Soft Policy Optimization (MASPO)**, a unified framework designed to harmonize these three dimensions. MASPO integrates a differentiable soft Gaussian gating to maximize gradient utility, a mass-adaptive limiter to balance exploration across the probability spectrum, and an asymmetric risk controller to align update magnitudes with signal confidence. Extensive evaluations demonstrate that MASPO serves as a robust, all-in-one RLVR solution, significantly outperforming strong baselines in sample efficiency and reasoning accuracy across diverse LLM scales (1.5B/7B/14B).

## 🎯 Key Contributions

- **Unified Perspective**: We systematically identify inherent challenges in current trust region paradigms and propose a holistic perspective that aligns RLVR optimization by addressing three fundamental misalignments: inefficient gradient utilization, probability mass insensitivity, and asymmetric signal reliability.

- **All-in-One Framework**: We propose a comprehensive solution, which unifies a soft Gaussian gating for continuous updates, a mass-adaptive limiter for targeted long-tail exploration, and an asymmetric risk controller for signal-aware optimization into a single framework.

- **Superior Performance**: Comprehensive evaluations on diverse mathematical benchmarks consistently demonstrate that MASPO achieves superior sample efficiency and reasoning performance. Further experiments confirm its robustness across varying LLM scales and effectiveness in stabilizing long-chain reasoning.

## 🚀 Quick Start

### Installation

This implementation is based on [verl](https://github.com/volcengine/verl), a flexible and efficient RLHF framework. Please follow the verl installation guide first.

```bash
# Clone the repository
git clone https://github.com/your-repo/MASPO-RL.git
cd MASPO-RL

# Install verl dependencies (see verl documentation for details)
pip install -r requirements.txt
```

### Running MASPO

We provide a complete example script for training with MASPO on GSM8K dataset:

```bash
bash examples/maspo_trainer/run_deepseek-r1-distill-qwen-7b.sh
```

### Key Configuration Parameters

To enable MASPO, set the following parameters in your configuration:

```yaml
actor_rollout_ref.actor.policy_loss.ratio_clip.ratio_mode: maspo
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_sigma_base: 1      # Base variance parameter
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_alpha: 0.3         # Mass-adaptive scaling factor
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_beta_pos: 0.03     # Positive risk control parameter
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_beta_neg: 0.03     # Negative risk control parameter
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_sigma_high: 10     # Upper bound for variance
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_adv_low: 0.1        # Lower bound for advantage scaling
actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_adv_high: 10       # Upper bound for advantage scaling
```

**Hyperparameter Guidelines:**
- **Default recommendation**: `maspo_sigma_base=1, maspo_alpha=0.3, maspo_beta_pos=0.03, maspo_beta_neg=0.03` (robust baseline)
- **For smaller models (1.5B)**: Can use `maspo_alpha=0.5` for better exploration in long-tail tokens
- **For larger models (7B+)**: Use `maspo_alpha=0.3` to maintain stability while preserving exploration
- **Mass-adaptive scaling (`maspo_alpha`)**: Controls how strongly the trust region adapts to token probability. Recommended range: `[0.3, 0.5]`
- **Risk control (`maspo_beta_pos`, `maspo_beta_neg`)**: Modulates update magnitude based on signal confidence. Recommended: `0.03` for both

### Example Configuration

Here's a minimal example showing how to configure MASPO:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    actor_rollout_ref.actor.policy_loss.ratio_clip.ratio_mode=maspo \
    actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_sigma_base=1 \
    actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_alpha=0.3 \
    actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_beta_pos=0.03 \
    actor_rollout_ref.actor.policy_loss.ratio_clip.maspo_beta_neg=0.03 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=15
```

### Full Training Script

For a complete training example, see `examples/maspo_trainer/run_deepseek-r1-distill-qwen-7b.sh`. This script includes:
- GRPO advantage estimator configuration
- MASPO policy loss configuration
- KL loss settings
- FSDP configuration
- vLLM rollout settings
- Wandb logging setup

## 📊 Experimental Results

### Main Results

We evaluate MASPO on multiple mathematical reasoning benchmarks including AIME24, AIME25, AMC23, MATH500, Minerva, and OlympiadBench. The following table shows the comprehensive comparison with competitive baselines:

#### DeepSeek-R1-Distill-Qwen-1.5B Results

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | **Avg.** |
|--------|--------|--------|-------|---------|---------|----------|----------|
|        | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 |
| GRPO | 33.2 / 71.8 | 27.7 / 49.9 | 79.5 / 94.8 | 77.6 / 90.8 | 26.1 / 48.8 | 46.3 / 64.7 | 48.4 / 70.1 |
| Clip Higher | 36.6 / 70.1 | 30.1 / 55.8 | 82.8 / 94.9 | 71.7 / 88.6 | 24.8 / 49.3 | 42.0 / 61.6 | 48.0 / 70.1 |
| DAC | 40.0 / 73.6 | 28.7 / 51.2 | 80.2 / 95.0 | 78.1 / 92.5 | 27.4 / 54.0 | 46.3 / 64.5 | 50.1 / 71.8 |
| Entropy Adv. | 29.6 / 67.8 | 23.9 / 44.2 | 77.1 / 94.1 | 78.5 / 89.7 | 25.2 / 48.6 | 46.4 / 64.3 | 46.8 / 68.1 |
| BAPO | 38.3 / 72.3 | 28.0 / 55.7 | 80.1 / 94.9 | 74.9 / 91.0 | 25.1 / 44.7 | 44.0 / 61.3 | 48.4 / 70.0 |
| SAPO | 39.3 / 73.7 | 28.0 / 49.9 | 82.7 / 94.8 | 79.1 / 91.4 | 29.6 / 52.7 | 48.2 / 66.7 | 51.2 / 71.5 |
| **MASPO (Ours)** | **41.0** / **74.8** | 28.4 / **58.0** | 82.2 / **95.0** | 78.0 / 89.7 | **30.7** / **54.1** | 47.8 / 65.7 | **51.4** / **72.9** |

#### DeepSeek-R1-Distill-Qwen-7B Results

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | **Avg.** |
|--------|--------|--------|-------|---------|---------|----------|----------|
|        | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 | A@32 / P@32 |
| GRPO | 48.2 / 82.5 | 37.4 / 60.5 | 88.1 / 96.6 | 84.8 / 92.4 | 37.4 / 57.2 | 57.2 / 73.9 | 58.9 / 77.2 |
| Clip Higher | 47.4 / 82.4 | 37.4 / 67.2 | 89.0 / 96.6 | 85.3 / 95.4 | 38.8 / 58.5 | 57.5 / 75.5 | 59.2 / 79.3 |
| DAC | 50.2 / 81.2 | 36.2 / 61.7 | 88.1 / 99.6 | 85.1 / 94.8 | 36.6 / 58.7 | 57.9 / 76.0 | 59.0 / 78.7 |
| Entropy Adv. | 49.1 / 81.5 | 34.4 / 58.1 | 87.7 / 96.6 | 84.8 / 92.5 | 36.5 / 54.0 | 56.4 / 73.7 | 58.2 / 76.1 |
| BAPO | 47.1 / 80.3 | 37.7 / 58.4 | 89.2 / 97.2 | 85.0 / 94.5 | 38.3 / 57.5 | 57.1 / 74.5 | 59.1 / 77.1 |
| SAPO | 47.5 / 79.2 | 35.3 / 58.0 | 88.7 / 95.0 | 85.5 / 92.9 | 39.4 / 58.1 | 56.6 / 74.6 | 58.8 / 76.3 |
| **MASPO (Ours)** | **53.2** / 82.4 | **42.9** / **73.2** | **91.4** / 95.0 | **86.0** / 94.7 | 39.3 / 58.6 | **58.0** / 74.9 | **61.8** / **79.8** |

**Key Findings:**
- **1.5B Model**: MASPO outperforms GRPO by **+3.0%** (48.4 → 51.4) and best baseline (SAPO) by **+0.2%** (51.2 → 51.4) in Avg@32
- **7B Model**: MASPO outperforms GRPO by **+2.9%** (58.9 → 61.8) and best baseline (Clip Higher) by **+2.6%** (59.2 → 61.8) in Avg@32
- MASPO demonstrates superior performance across the majority of benchmarks on both scales
- **Bold** indicates best performance, underlined represents second-best

### Scalability Analysis

| Method | 1.5B (A@32 / P@32) | 7B (A@32 / P@32) | 14B (A@32 / P@32) |
|--------|---------------------|-------------------|-------------------|
| GRPO | 48.4 / 70.1 | 58.9 / 77.2 | 53.6 / 67.4 |
| **MASPO** | **51.4 / 72.9** | **61.8 / 79.8** | **56.4 / 71.1** |
| *Improvement* | *+3.0 / +2.8* | *+2.9 / +2.6* | *+2.8 / +3.7* |

MASPO demonstrates consistent improvements across all model scales, confirming its scalability and robustness.

## 🔬 Algorithm Overview

### Core Idea

MASPO addresses three fundamental misalignments in current RLVR paradigms by proposing a unified framework:

1. **Inefficient Gradient Utilization**: Hard clipping imposes a binary cutoff that discards valuable directional gradients from exploratory samples exceeding the boundary, thereby significantly diminishing the effective utilization of informative gradient signals.

2. **Probability Mass Insensitivity**: Uniform ratio constraints ignore the vast disparity in token probabilities, failing to account for the massive mass displacement in head tokens versus the negligible shift in tail tokens.

3. **Asymmetric Signal Reliability**: Symmetric advantage handling ignores the disparate signal-to-noise ratios between verified positive solutions and ambiguous negative ones.

### Mathematical Formulation

The MASPO framework integrates three key components:

#### Soft Gaussian Gating

MASPO replaces hard clipping with a differentiable soft Gaussian gating mechanism:

$$\mathcal{F}^\text{MASPO}_{i,t} = \begin{cases}
\exp \left( {-\frac{(\rho_{i,t}(\theta)-1)^{2}}{2 \sigma^{2}_\text{pos}}} \right) & \text{if } \hat{A}_{i,t}>0 \land \rho_{i,t}(\theta)>1 \\
\exp \left( {-\frac{(\rho_{i,t}(\theta)-1)^{2}}{2 \sigma^{2}_\text{neg}}} \right) & \text{if } \hat{A}_{i,t}<0 \land \rho_{i,t}(\theta)<1 \\
1, & \text{otherwise}
\end{cases}$$

#### Dual-Variable Adaptive Variance

The variance $\sigma$ is dynamically determined by combining mass-adaptive scaling and asymmetric risk control:

$$\sigma_\text{pos} = \underbrace{\frac{\sigma_\text{base}}{\pi_{\theta_{old}}^{\alpha}}}_{\text{Mass-Adaptive}} \cdot \underbrace{\left( 1+\beta_\text{high} \hat{A}_{i,t} \right)}_{\text{Risk Controller}}$$

$$\sigma_\text{neg} = \underbrace{\frac{\sigma_\text{base}}{\pi_{\theta_{old}}^{\alpha}}}_{\text{Mass-Adaptive}} \cdot \underbrace{\left(1-\beta_\text{low} \hat{A}_{i,t} \right)^{-1}}_{\text{Risk Controller}}$$

Where:
- **Mass-Adaptive Limiter**: The term $\frac{\sigma_\text{base}}{\pi_{\theta_{old}}^{\alpha}}$ inversely scales the trust region width with token probability, expanding exploration budget for low-probability tokens while enforcing strict constraints for high-probability tokens.
- **Asymmetric Risk Controller**: The second component modulates the trust region based on signal confidence, expanding for high-confidence positive signals and constraining for ambiguous negative signals.

### Key Advantages

1. **Continuous Optimization Landscape**: Soft Gaussian gating converts the disjoint cliff (characteristic of binary clipping) into a smooth and continuous manifold, ensuring samples that marginally exceed the trust region contribute attenuated but non-zero gradients.

2. **Targeted Exploration**: Mass-adaptive scaling allocates exploration budget dynamically based on token probability mass, enabling efficient exploration in long-tail regions while maintaining stability in high-probability regions.

3. **Signal-Aware Updates**: Asymmetric risk control aligns update magnitudes with signal confidence, maximizing utilization of verified positive signals while preventing catastrophic unlearning from ambiguous negative signals.

## 📁 Project Structure

```
maspo/
├── verl/
│   ├── trainer/ppo/
│   │   └── core_algos.py          # MASPO implementation
│   ├── workers/actor/
│   │   └── dp_actor.py            # Actor with MASPO support
│   └── trainer/config/actor/
│       └── actor.yaml             # Configuration file
├── examples/
│   └── maspo_trainer/
│       └── run_deepseek-r1-distill-qwen-7b.sh        # Example training script
└── README.md
```

## 🔧 Implementation Details

### Code Location

- **Core Algorithm**: `verl/trainer/ppo/core_algos.py` (lines 907-922)
- **Actor Integration**: `verl/workers/actor/dp_actor.py` (uses `compute_policy_loss` with MASPO mode)
- **Configuration**: `verl/trainer/config/actor/actor.yaml` (ratio_clip section)

### Key Implementation Points

1. MASPO is integrated into the existing PPO framework via the `ratio_clip_config.ratio_mode` parameter
2. When `ratio_mode="maspo"`, the algorithm applies the soft Gaussian gating with dual-variable adaptive variance
3. The implementation supports both FSDP and Megatron backends
4. The unilateral design selectively attenuates aggressive overshoots without hindering conservative updates

## 🔍 Related Work

This work builds upon and improves several existing methods:

- **GRPO** (Group Relative Policy Optimization): The baseline hard-clipping method
- **SAPO** (Soft Advantage Policy Optimization): Soft clipping with global gating mechanism
- **DAC** (Dynamic Adaptive Clipping): Adapts bounds based on policy probabilities
- **BAPO** (Balanced Advantage Policy Optimization): Dynamically adjusts clipping bounds to ensure positive sample contribution
- **Clip Higher**: Relaxes upper bounds globally but ignores token-specific probabilities
- **Entropy Advantage**: Adds entropy terms to the advantage for implicit rebalancing

For more details on the theoretical analysis and comparison, please refer to our paper.

## 🙏 Acknowledgments

This implementation is built on top of [verl](https://github.com/volcengine/verl), a flexible and efficient RLHF framework. We thank the verl community for their excellent infrastructure.

## 🐛 Troubleshooting

### Common Issues

1. **Training Instability**: If you encounter training collapse, try reducing `maspo_alpha` or `maspo_beta_pos`/`maspo_beta_neg` values. For larger models, use more conservative settings (e.g., `alpha=0.3, beta=0.03`).

2. **Memory Issues**: Ensure you have sufficient GPU memory. Consider using gradient checkpointing and parameter offloading:
   ```yaml
   actor_rollout_ref.model.enable_gradient_checkpointing: True
   actor_rollout_ref.actor.fsdp_config.param_offload: True
   ```

3. **Convergence Speed**: If training is too slow, you can increase the learning rate slightly, but monitor for stability. MASPO's soft gating mechanism generally allows for more aggressive learning rates compared to hard clipping.

### Performance Tips

- Use `use_remove_padding=True` for better memory efficiency
- Enable `use_fused_kernels=True` if your model supports it
- Adjust `ppo_mini_batch_size` and `ppo_micro_batch_size_per_gpu` based on your GPU memory
- For long-tail exploration, consider increasing `maspo_alpha` to 0.5 for smaller models

### Hyperparameter Tuning

- **Mass-Adaptive Scaling (`maspo_alpha`)**: 
  - Lower values (0.1-0.2): More conservative, similar to GRPO behavior
  - Recommended range: 0.3-0.5 for balanced exploration and stability
  - Higher values (0.8+): May cause instability, use with caution

- **Risk Control (`maspo_beta_pos`, `maspo_beta_neg`)**:
  - Default: 0.03 for both parameters
  - Higher values: More aggressive signal-aware updates
  - Lower values: More conservative, closer to uniform handling

## 📝 License

This project follows the same license as verl. Please refer to the verl repository for license details.

---

**Note**: This is the official implementation of MASPO. For more details, please refer to our paper.
