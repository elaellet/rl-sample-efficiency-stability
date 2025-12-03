# Sample Efficiency and Stability Analysis of SAC+HER on panda-gym

## Overview
This project investigates the sample efficiency and stability of **Soft Actor-Critic (SAC)** combined with **Hindsight Experience Replay (HER)** on the `PandaPush` task.

Reinforcement Learning faces two significant hurdles: **sample efficiency** and **stability**. While reward shaping (Dense Rewards) is often used to accelerate learning, it requires extensive domain knowledge and can introduce bias and local optima. This study aims to determine if learning from sparse, binary rewards using HER can offer a more robust alternative to dense, shaped rewards.

The project compares three configurations:
1. **SAC+Dense**: Standard SAC using a shaped, dense reward function (negative Euclidean distance).
2. **SAC+Sparse**: Standard SAC using a binary, sparse reward function.
3. **SAC+HER+Sparse**: SAC augmented with HER using a sparse reward function.


## Structure
```
├── hyperparams/
├── logs/
│   ├── sac/
├── notebooks/
│   ├── environment_exploration.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── analyze.py
│   ├── evaluate.py
│   ├── policies.py
│   ├── train.py
│   ├── utils.py
│   ├── visualizations.py
│   ├── visualize_landscape.py
├── .gitignore
├── REAMD.md
└── requirements.txt
```


## Experimental Setup

### Environment & Implementation
- **Task**: `PandaPushJoints-v3` (Sparse) & `PandaPushJointsDense-v3` (Dense).
- **Goal**: The robotic arm must push an object to a target location.
- **Library**: Stable-Baselines3 (SB3) & RL Baselines3 Zoo.
- **Seeds**: 5 random seeds (`42`, `43`, `44`, `45`, `46`) were used for each configuration to ensure statistical significance.
- **Hyperparameters**: Identical across all configurations (derived from original SAC/HER papers) to isolate the algorithmic impact. No reward normalization was applied to the sparse environment.

### Reproduction
**1. Train**

To train an agent using the `train.py` script from RL Baselines3 Zoo.
```
python -m rl_zoo3.train --algo sac --env PandaPushJointsDense-v3 -n 1000000 --eval-freq 25000 --eval-episodes 20 --n-eval-envs 1 --save-freq 100000 -f logs/ --seed 42 -conf hyperparams/sac.yml
```

**2. Analyze**

To generate the statistical summary and learning curves from the training logs:
```
python -m src.analyze --env-id PandaPushJointsDense-v3 --algo sac --log-folder ./logs/sac/sac_dense_push --plot-filename learning_curves_sac_dense_push
```

**3. Evaluate**

To evaluate trained agents:

**Individual Seed:** To visualize a single trained agent and calculate the success rate:
```
python -m src.evaluate --env-id PandaPushJointsDense-v3 --algo sac --seed 42 --log-folder ./logs/sac/sac_dense_push_42
```

**Batch Mode (Aggregated Stats):** To evaluate all seeds in a folder and generate a stability boxplot:
```
python -m src.evaluate --env-id PandaPushJointsDense-v3 --algo sac --batch-folder ./logs/sac
```

**4. Analyze Value Landscape**

To generate the "Inconsistent Landscape" plots comparing Dense vs. Sparse (HER) critics:
```
python -m src.visualize_landscape --env-id PandaPushJoints-v3 --seed 42 --dense-folder ./logs/sac/sac_dense_push_42 --her-folder ./logs/sac/sac_her_sparse_push_42
```

## Experimental Results
### 1. Learning Curves
The following plots illustrate the success rate over 1 million timesteps. Shaded regions represent the variability across the 5 seeds.

- **SAC+Dense**: Demonstrated significant **instability**, indicated by the wild fluctuations in the learning curves. While some seeds converged quickly, others (Seed 43, 45) failed to reach the threshold or exhibited high variance.
- **SAC+Sparse**: Failed to learn. Due to the sparsity of the reward signal and the lack of HER, the agent never discovered the goal signal.
- **SAC+HER+Sparse**: Demonstrated superior **training stability**. All 5 seeds converged to a 100% success rate. However, Seed 43 was an outlier in terms of speed, taking significantly longer to solve the task, which contributed to high variance in the mean timesteps metric.

### 2. Quantitative Analysis
A "Converged Run" is defined as a run that reaches a 90% success rate threshold within the time limit.

| Metric | SAC (Dense) | SAC (Sparse) | SAC+HER (Sparse) |
| :---: | :---: | :---: | :---: |
| **Total Runs** | 5 | 5 | 5 |
| **Converged Runs** | 3/5 (60.0%) | 0/5 (0.0%) | 5/5 (100.0%) |
| **Mean Timesteps** | 766,666 ± 31,180 | N/A | 490,000 ± 188,812 |
| **Mean Best Success** | 96.00% ± 5.83% | 82.00% ± 7.48% | 100.00% ± 0.00% |
| **Mean Norm. AUC** | 0.50 ± 0.12 | 0.33 ± 0.06 | 0.77 ± 0.13 |

**Key Finding**: SAC+HER was the only configuration to achieve **100% convergence**. While SAC+Dense failed in 40% of runs, SAC+HER reliably solved the task every time, albeit with high variance in the time required (Mean ~490k steps).

### 3. Final Policy Evaluation
After training, the best model from each seed was evaluated over 200 episodes.
- **SAC+Dense Agg**: 90.10% ± 8.19%
- **SAC+Sparse Agg**: 73.30% ± 11.60%
- **SAC+HER+Sparse Agg**: 80.80% ± 11.17%

While HER showed perfect training convergence, the final evaluation indicates that `SAC+Dense` policies—when they succeeded—generalized slightly better on average. However, the Dense configuration remains prone to total failure on specific seeds, whereas HER produced a viable policy in every single run.

## Analysis: The "Inconsistent Landscape" Hypothesis
To understand why **SAC+Dense** is unstable despite having a shaped reward, I analyzed the **Value Function Landscape** by querying the critic at various distances from the goal. This revealed that Dense rewards create highly unpredictable value landscapes, whereas HER provides geometric consistency.

### 1. HER: The Robust  Gradient
In all seeds (42-46), the **SAC+HER** critic (Blue Line) learned a steep, linear, and monotonic value function. It correctly learned that "Closer = Higher Value," providing a robust geometric signal to the actor regardless of the initialization or training duration.

### 2. Dense: The Unreliable Gradients
The Dense critic (Red Line) consistently failed to model the task geometry reliably. Even among the runs that technically "succeeded" (converged to high success rates), the underlying value landscapes varied drastically.

- **Mode A: The Misleading Gradient (Seed 43 - Failed)** The value function is non-monotonic. Between 0cm and 20cm, the value curve rises incorrectly. Rather than rewarding the agent for approaching the goal (0cm), the critic incorrectly assigns a higher value to being 20cm away (~-8.19) than being at the goal (~-8.28). This creates a local optimum "trap" at 20cm, preventing the agent from stabilizing at the target.

- **Mode B: The Vanishing Gradient (Seed 45 - Failed)** The value function has the correct direction but an extremely weak slope. The value difference between the start (30cm) and the goal (0cm) is negligible (Delta ~0.25). This signal is likely too faint to overcome entropy regularization, causing the agent to drift rather than converge.

- **Mode C: Lucky vs. Valid Convergence (Seed 44 vs. Seed 46)** The most critical insight comes from comparing the two successful runs. Despite both achieving ~100% success, their underlying mechanics were fundamentally different:

  - **The Lucky Success (Seed 44)**: The critic suffered from **Gradient Inversion**. It learned that being far away (30cm) is better than being close (0cm). The agent solved the task in spite of the critic, likely through stochastic trajectory memorization rather than geometric understanding.
  - **The Valid Success (Seed 46)**: This was the sole instance where the Dense reward worked as intended. The critic learned a correct, monotonic gradient (dropping from -8.17 to -8.8).

* **Implication**
The comparison between Seed 44 and Seed 46 proves that **convergence in SAC+Dense is unreliable**. A high success rate does not guarantee a healthy policy; the agent might be succeeding by luck with a broken critic (Seed 44) or by genuine learning (Seed 46). In contrast, HER guarantees that if the agent converges, it does so with a mathematically correct understanding of the task geometry.

## Reflection
The results highlight a critical trade-off in Reinforcement Learning design:

1. **Reward Shaping is Unreliable**: While Dense rewards are intended to help, they produce inconsistent value landscapes. Even when agents converge to high success rates, the underlying learning is unpredictable: one agent might learn correctly (Seed 46), while another succeeds "by luck" with an inverted critic (Seed 44). This makes Dense rewards deceptive—metrics alone cannot distinguish between robust learning and stochastic overfitting.

2. **HER provides Geometric Truth**: By learning from sparse outcomes, HER forces the critic to learn the true probability of success. This results in a stable, interpretable value landscape that guarantees convergence (100% in this study).

3. **The Efficiency Gap**: Despite its stability, HER remains sample inefficient (Mean ~490k steps).


## Limitations
1. **Sample Efficiency**: While HER solves the stability issue, it remains computationally expensive. The SAC+HER configuration required a mean of **~490,000 timesteps** to solve a simple push task, making it potentially prohibitive for real-world robotics where interaction time is costly.

2. **Hyperparameter Brittleness (Theoretical)**: For SAC+HER, I adhered to the hyperparameters derived from the original paper to ensure stability. However, I suspect the method retains the high sensitivity to configuration typical of off-policy algorithms. My experience tuning the **SAC+Dense** baseline revealed that performance was extremely brittle regarding hyperparameter choices. It is highly likely that SAC+HER shares this characteristic; deviating from the reference parameters would likely cause the performance to collapse.

3. **Initialization Variance**: The high variance in convergence speed indicates that the method is still sensitive to initialization, even if it eventually guarantees success. For example, while Seed 46 converged rapidly in **325,000 steps**, Seed 45 required **850,000 steps**—more than double the training time—to reach the same threshold. This unpredictability in training duration can be problematic for scheduling resources in large-scale experiments.

## Acknowledgments
- **Soft Actor-Critic**: Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.

- **Hindsight Experience Replay**: Andrychowicz, M., et al. (2017). *Hindsight Experience Replay*.

- **Books**: 
  - *Reinforcement Learning: An Introduction (Sutton & Barto)*
  - *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)*

- **Resources**: OpenAI Spinning Up