Here is a ranked list of 100 AI safety techniques and methods used in or alongside RLHF, organized by category and ranked roughly from foundational/high-impact to niche/emerging. The ranking reflects a combination of empirical evidence, adoption in frontier labs, and theoretical promise.

---

### 🛡️ Core Safe RLHF & Constrained Optimization (1–20)

1. **Safe RLHF** – Decouples helpfulness and harmlessness into separate reward and cost models, solved via Lagrangian methods .
2. **PPO-Lagrangian (PPO-Lag)** – Standard PPO with a Lagrangian multiplier on cost constraints .
3. **SafeDPO** – Lightweight safety-constrained DPO preserving the optimal solution with one extra hyperparameter .
4. **NSPO (Null-Space Constrained Policy Optimization)** – Projects safety gradients into the null space of general tasks to mitigate the alignment tax .
5. **CS-RLHF (Certifiable Safe RLHF)** – Fixed-penalty constraint optimization with semantic grounding .
6. **HC-RLHF (High-Confidence Safe RLHF)** – Provides high-confidence safety guarantees via upper-confidence bounds on cost .
7. **Safe RLHF-V** – First multimodal safety alignment framework with dual preference annotations .
8. **RePO (Rectified Policy Optimization)** – Replaces average expected-cost constraint with per-sample rectified cost penalties .
9. **FOCOPS** – First-order constrained optimization integrated into RLHF for safe language generation .
10. **P3O** – Penalized Proximal Policy Optimization, showing strong potential for safe language generation .
11. **Primal-Dual DPO** – Provably convergent primal-dual method for constrained LLM alignment .
12. **CAID (Constrained Alignment via Iterative Dualization)** – Iterative dualization alternately updates policy and dual variables .
13. **RAD (Risk-sensitive Alignment via Dominance)** – Replaces scalar expected-cost constraints with First-Order Stochastic Dominance constraints .
14. **Optimistic Primal-Dual (OPD)** – Universal primal-dual framework with predictive updates for stable saddle-point dynamics .
15. **ACPO (Average-Constrained Policy Optimization)** – Constrained MDP under average-cost criterion .
16. **CGPO (Constrained Generative Policy Optimization)** – Integrates trajectory, action, cost, and output constraints .
17. **Offline Constrained RLHF** – Constrained RLHF with multiple preference oracles in offline settings .
18. **PSPL** – Preference-based RL with joint sampling of reward models and transition dynamics .
19. **MOPO** – Multi-objective preference optimization for safe alignment .
20. **DPOC (Direct Preference Optimization with Pareto Dominance Constraint)** – Constrained multi-objective alignment with wider Pareto coverage .

---

### 🔬 Reward Model Improvements & Debiasing (21–40)

21. **RuleAdapter** – Dynamically selects the five most important safety rules per response pair .
22. **BNRM (Bayesian Non-Negative Reward Model)** – Integrates non-negative factor analysis into Bradley–Terry modeling .
23. **ENCORE** – Entropy-guided reward composition for multi-head safety reward models .
24. **GUARD / Fair-RM** – Minimizes mutual information between reward scores and sensitive categories via adversarial training .
25. **DIR (Debiasing via Information optimization for RM)** – Information-theoretic debiasing method for reward models .
26. **Post-hoc Reward Calibration** – Corrects biases (e.g., length bias) without additional training .
27. **BSR (Batch-wise Sum-to-zero Regularization)** – Enforces zero-centered rewards per batch to constrain abnormal magnitudes .
28. **BSPO (Behavior-Supported Policy Optimization)** – Regularizes value function to penalize OOD values .
29. **T-REG (Token-Level Reward Regularization)** – Uses both sequence-level and token-level rewards for preference optimization .
30. **CausalRM** – Causal-theoretic reward modeling to mitigate spurious correlations .
31. **Graph-Preference Learning** – Debiases network-sampled human feedback for target welfare estimation .
32. **UARD (Uncertainty-Aware Reward Discounting)** – Reduces reward contribution when ensemble disagrees or annotator variance is high .
33. **BNRM (Bayesian Non-negative Reward Modeling)** – Sparse latent factors act as implicit debiasing .
34. **TLCR (Token-Level Continuous Reward)** – Discriminator-based token-level reward for RLHF .
35. **ArmoRM** – Multi-objective reward model with explicit objective separation .
36. **MODPO (Multi-Objective DPO)** – Extends DPO to multiple objectives including safety .
37. **MinMaxRLHF** – Addresses differing annotator preferences by minimizing worst-case regret .
38. **Bi-Factorial Preference Optimization (BFPO)** – Re-parameterizes joint RLHF safety-helpfulness objective into supervised learning .
39. **BFPO (Bi-Factorial Preference Optimization)** – Labeling function captures global preference ranking to balance safety and helpfulness .
40. **Preference As Reward (PAR)** – Leverages latent preferences embedded within the reward model as RL signal .

---

### 🚀 Reward Hacking Mitigation (41–60)

41. **Gradient Regularization (GR)** – Biases policy updates toward flatter regions where reward is more accurate .
42. **EPPO (Energy Loss-Aware PPO)** – Penalizes excessive energy loss in the final layer to mitigate reward hacking .
43. **SignCert-PO (Sign-Certified Policy Optimization)** – Down-weights non-robust completions in policy gradient updates .
44. **PET (Pessimistic Reward Fine-Tuning)** – Learns pessimistic reward model robust against reward hacking .
45. **χ² Divergence Regularization** – Regularizes occupancy measures instead of action distributions .
46. **ODIN (Disentangled Reward)** – Disentangles reward into helpfulness and safety components to reduce hacking .
47. **Reward Clipping** – Explicitly clips reward values to prevent extreme optimization .
48. **Length Penalty** – Penalizes longer responses to prevent length-based reward hacking .
49. **Reward Shaping (Bounded & Smoothed)** – Upper-bounds reward with rapid initial growth followed by gradual convergence .
50. **Behavior-Supported Regularization** – Penalizes OOD values without impacting in-distribution ones .
51. **Uncertainty-Penalized RLHF (UP-RLHF)** – Incorporates uncertainty penalties with diversified LoRA ensembles .
52. **Reward Model Ensemble (WCO)** – Worst-case optimization across multiple reward models .
53. **Reward Model Ensemble (UWO)** – Uncertainty-weighted optimization across ensemble .
54. **Reward Model Retraining** – Periodically retrains reward model on latest model outputs .
55. **IR³ (Contrastive Inverse RL)** – Four surgical mitigation strategies: clean reward optimization, adversarial shaping, constrained optimization, feature-guided distillation .
56. **Adversarial Reward Auditing** – Active detection and mitigation of reward hacking .
57. **Trajectory-Level Behavior Monitoring** – Monitors behavior during RL to mitigate policy-dependent shortcut exploitation .
58. **Hybrid Verifiers** – Combines multiple verification approaches to reduce hacking .
59. **Structured Rubrics** – Decomposes reward into explicit, orthogonal components .
60. **Gated Reward Accumulation (G-RA)** – Gated accumulation to prevent reward hacking .

---

### 🧠 Process Reward Models & Step-Level Supervision (61–72)

61. **Process Reward Models (PRMs)** – Provides step-level feedback on intermediate reasoning steps .
62. **AURA** – Uses PRMs to assess logical coherence and safety-awareness at each step .
63. **STAIR** – Constructs preference pairs with step-wise reward function encouraging safe and helpful answers .
64. **ERPO** – Three-level ranking: helpful reason + safe answer > harmful prefix + self-reflection > incorrect reason + harmful answer .
65. **SaRO** – Decomposes thinking chain into steps, encourages early reflection with fewer unsafe steps .
66. **RATIONAL** – Prompts model to evaluate safety of query and reject malicious intent .
67. **R2D** – Safety reasoning suffix appended to every reasoning step (SAFE/UNSAFE/RETHINK) .
68. **S2T-RLHF** – Hierarchical credit assignment for stable preference-based RLHF .
69. **Rule-Based Rewards (RBR)** – Uses explicit rules for reward assignment .
70. **Fine-Grained RLHF** – Provides fine-grained feedback at multiple levels .
71. **Hierarchical RLHF** – Four-tier rules (invariant/locked/watched/free) with per-tier reward models .
72. **Quantile-Guided Alignment (QA)** – Users specify desired improvements at any quantile across multiple reward dimensions .

---

### ⚡ Inference-Time Alignment & Decoding (73–84)

73. **LARA (Lagrangian Reward Augmentation)** – Decoder-agnostic framework for safe inference-time alignment .
74. **SITAlign** – Satisficing alignment at inference-time with threshold-based constraints .
75. **RAIN (Rewindable Auto-regressive INference)** – Allows pre-trained LLMs to evaluate and rewind generation for safety .
76. **DSA (Disentangled Safety Adapters)** – Decouples safety computations from task-optimized base model .
77. **SafeInfer** – Context-adaptive decoding-time safety alignment .
78. **BlendIn** – Quality-aware probabilistic distribution blending for inference-time alignment .
79. **GGRO (Gradient-Guided Reward Optimization)** – Improves inference-time alignment across safety and helpfulness .
80. **ALIGNBEAM** – Cross-vocabulary logit mixing for inference-time alignment transfer .
81. **Speculative Safety-Aware Decoding (SSD)** – Safety-aware speculative decoding .
82. **InfAlign** – Inference-aware language model alignment .
83. **STARS** – Synchronous token alignment for robust supervision .
84. **Nudging** – Guided decoding for inference-time alignment .

---

### 🤝 Multi-Agent Debate & Verification (85–92)

85. **MADRA (Multi-Agent Debate for Risk-Aware Embodied Planning)** – Training-free cognitive architecture mimicking System-2 deliberation .
86. **Multi-Agent Debate with Memory** – Peers critique each other yielding richer safety feedback .
87. **Multi-Agent RL for Provably Robust LLM Safety** – Achieves 95% reduction in harmful outputs with only 5% increase in refusal rates .
88. **Aetheria** – Multi-agent debate framework with strong safety detection capabilities .
89. **Critic Councils** – Multi-agent collaboration for safe and robust RLHF .
90. **Consensus-Based Reward** – Framework for mitigating malicious RLHF feedback .
91. **Multi-Agent Constitutional Debate** – Three AIs debate under a constitution .
92. **MESA (Decentralized Expertise)** – Reformulates MoE safety alignment as resource allocation across experts .

---

### 📜 Constitutional AI, RLAIF & Principle-Based (93–100)

93. **Constitutional AI (CAI)** – Two-phase approach: supervised self-critique/revision, then RLAIF .
94. **RLAIF (Reinforcement Learning from AI Feedback)** – Replaces human feedback with AI-generated feedback .
95. **Contextual Constitutional AI** – Variation of CAI that considers context when deciding response pairs .
96. **Reverse Constitutional AI (R-CAI)** – Inverts CAI principles into a "Toxic Constitution" for adversarial data generation .
97. **QA-LIGN** – Replaces opaque scalar rewards with interpretable, principle-based decomposed QA .
98. **Deliberative Alignment** – Model reasons through safety specifications before responding .
99. **IterAlign** – Iterative alignment using constitutional principles .
100. **ICAI (Iterative Constitutional AI)** – Iterative refinement against constitutional principles .

---

### 💎 Summary

This ranked list spans **constrained optimization**, **reward model engineering**, **reward hacking defenses**, **process supervision**, **inference-time control**, **multi-agent debate**, and **principle-based alignment**. The top-ranked methods (Safe RLHF, NSPO, Gradient Regularization, RuleAdapter, CAI/RLAIF) are those with the strongest empirical evidence and widest adoption. Lower-ranked entries represent promising but less-validated directions. No single technique is sufficient; robust safety requires **layered, hybrid approaches** combining training-time constraints, reward model robustness, inference-time guardrails, and continuous oversight.









/////////////////////////////////////////////////////////////////////////////////////////////
# 100 RLHF-Related AI Safety Methods

One line per method. Grouped loosely by theme; numbering is continuous 1–100.

## Foundational RLHF & Instruction-Tuning (1–10)
1. **Deep RL from Human Preferences** (Christiano et al., 2017) — the original RLHF paper: agents learn from pairwise trajectory comparisons instead of a hand-written reward. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
2. **Fine-Tuning Language Models from Human Preferences** (Ziegler et al., 2019) — first application of the RLHF reward-model + PPO recipe to text generation. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
3. **Learning to Summarize from Human Feedback** (Stiennon et al., 2020) — reward model trained on human comparisons, then PPO fine-tuning for summarization. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
4. **WebGPT** (Nakano et al., 2021) — RLHF for browser-assisted, citation-grounded question answering. [arXiv:2112.09332](https://arxiv.org/abs/2112.09332)
5. **InstructGPT** (Ouyang et al., 2022) — RLHF for instruction-following; the direct basis for ChatGPT. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
6. **Training a Helpful and Harmless Assistant with RLHF** (Bai et al., 2022) — Anthropic's HH-RLHF dataset and dual helpfulness/harmlessness objective. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
7. **Sparrow** (Glaese et al., 2022) — DeepMind dialogue agent trained with rule-conditional reward models. [arXiv:2209.14375](https://arxiv.org/abs/2209.14375)
8. **Teaching LMs to Support Answers with Verified Quotes** (Menick et al., 2022) — RLHF combined with retrieval for factual grounding. [arXiv:2203.11147](https://arxiv.org/abs/2203.11147)
9. **Illustrating RLHF** — widely used plain-language explainer of the standard reward-model + PPO pipeline. [Hugging Face blog](https://huggingface.co/blog/rlhf)
10. **Awesome-RLHF** — continuously updated curated index of RLHF research papers. [GitHub](https://github.com/opendilab/awesome-RLHF)

## Reward Modeling, Data & Uncertainty (11–18)
11. **Scaling Laws for Reward Model Overoptimization** (Gao et al., 2022) — quantifies Goodharting/reward-hacking as policies over-optimize a learned reward. [arXiv:2205.09235](https://arxiv.org/abs/2205.09235)
12. **Reward Model Ensembles Help Mitigate Overoptimization** — uses ensembling and uncertainty to curb reward hacking. [arXiv:2310.02743](https://arxiv.org/abs/2310.02743)
13. **Uncertainty Estimation for Language Reward Models** (Gleave & Irving, 2022) — conservative reward modeling under epistemic uncertainty. [arXiv:2203.07472](https://arxiv.org/abs/2203.07472)
14. **RewardBench** (Lambert et al., 2024) — standardized benchmark for evaluating reward-model quality. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
15. **RRM: Robust Reward Model Training Mitigates Reward Hacking** (Liu et al., 2024) — removes spurious correlations from reward-model training. [arXiv:2409.13156](https://arxiv.org/abs/2409.13156)
16. **Active Preference Optimization for Sample-Efficient RLHF** (Das et al., 2024) — active learning to reduce the cost of preference labeling. [arXiv:2402.10500](https://arxiv.org/abs/2402.10500)
17. **Theoretical Guarantees on the Best-of-N Alignment Policy** (Beirami et al., 2024) — formal analysis of best-of-N sampling as a lightweight alignment method. [arXiv:2401.01879](https://arxiv.org/abs/2401.01879)
18. **Awesome LLM Human-Preference Datasets** — curated list of datasets used to train RLHF/DPO reward and preference models. [GitHub](https://github.com/glgh/awesome-llm-human-preference-datasets)

## Direct Alignment / Preference-Optimization Algorithms (19–32)
19. **Direct Preference Optimization (DPO)** (Rafailov et al., 2023) — reward-model-free RLHF via a closed-form policy objective. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
20. **ΨPO / Identity Preference Optimization** (Azar et al., 2023) — general theoretical paradigm unifying RLHF and DPO. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
21. **SLiC-HF: Sequence Likelihood Calibration** (Zhao et al., 2023) — hinge-loss calibration between preferred/dispreferred sequences. [arXiv:2305.10425](https://arxiv.org/abs/2305.10425)
22. **KTO: Prospect-Theoretic Alignment** (Ethayarajh et al., 2024) — preference optimization from unpaired binary (good/bad) feedback. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
23. **RRHF: Rank Responses Without Tears** (Yuan et al., 2023) — ranking-loss alternative to PPO-based RLHF. [arXiv:2304.05302](https://arxiv.org/abs/2304.05302)
24. **ORPO: Monolithic Preference Optimization** (Hong et al., 2024) — merges SFT and preference tuning into a single reference-free loss. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
25. **SimPO: Simple Preference Optimization** (Meng et al., 2024) — reference-free, length-normalized DPO variant. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
26. **R-DPO: Disentangling Length from Quality** (Park et al., 2024) — corrects length bias in preference optimization. [arXiv:2403.19159](https://arxiv.org/abs/2403.19159)
27. **Generalized Preference Optimization (GPO)** (Tang et al., 2024) — unifying family of offline alignment losses. [arXiv:2402.05749](https://arxiv.org/abs/2402.05749)
28. **RainbowPO** — combines the most effective components of prior DPO variants into one method. [arXiv:2410.04203](https://arxiv.org/abs/2410.04203)
29. **Token-Level Direct Preference Optimization** — extends DPO from sequence-level to per-token credit assignment. [arXiv:2404.11999](https://arxiv.org/abs/2404.11999)
30. **Diffusion-DPO** — extends direct preference optimization to diffusion (image-generation) models. [arXiv:2311.12908](https://arxiv.org/abs/2311.12908)
31. **Is DPO Superior to PPO for LLM Alignment?** (Xu et al., 2024) — controlled empirical comparison of the two paradigms. [arXiv:2404.10719](https://arxiv.org/abs/2404.10719)
32. **Self-Play Fine-Tuning (SPIN)** (Chen et al., 2024) — converts a weak SFT model into a strong one via self-play preference data. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)

## Core RL Optimizers Used Inside RLHF (33–37)
33. **Proximal Policy Optimization (PPO)** (Schulman et al., 2017) — the RL optimizer used in most classic RLHF pipelines. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
34. **Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015) — PPO's stability-constrained predecessor. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
35. **DeepSeekMath / Group Relative Policy Optimization (GRPO)** (Shao et al., 2024) — critic-free RL objective for reasoning models. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
36. **DeepSeek-R1** — GRPO-driven RL used to incentivize reasoning capability at scale. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
37. **ReST: Reinforced Self-Training** (Gulcehre et al., 2023) — offline, growing-batch RL from model-generated data. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)

## Constitutional AI, RLAIF & Self-Improvement (38–45)
38. **Constitutional AI** (Bai et al., 2022) — self-critique/revision plus RL from AI Feedback (RLAIF), guided by written principles. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
39. **RLAIF vs. RLHF** (Lee et al., 2023) — Google study showing AI-labeled preferences can match human-labeled ones. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
40. **Self-Critiquing Models for Assisting Human Evaluators** (Saunders et al., 2022) — models write critiques of their own outputs to aid oversight. [arXiv:2206.05802](https://arxiv.org/abs/2206.05802)
41. **Self-Refine** (Madaan et al., 2023) — iterative self-feedback and revision without any extra training. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
42. **Self-Rewarding Language Models** (Yuan et al., 2024) — the policy model also acts as its own reward model (LLM-as-a-judge). [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
43. **RAIN** — self-alignment via self-evaluation and "rewind," requiring no fine-tuning. [arXiv:2309.07124](https://arxiv.org/abs/2309.07124)
44. **Aligner** (Ji et al., 2024) — a small model learns to correct a larger model's outputs post hoc. [arXiv:2402.02416](https://arxiv.org/abs/2402.02416)
45. **Awesome-RLAIF** — curated index of RL-from-AI-Feedback research. [GitHub](https://github.com/vicgalle/awesome-rlaif)

## Safety-Constrained & Harm-Reduction Variants (46–53)
46. **Safe RLHF** (Dai et al., 2023) — Lagrangian constrained optimization with a separate cost model for harmfulness. [arXiv:2310.12773](https://arxiv.org/abs/2310.12773)
47. **Towards Understanding Sycophancy in Language Models** (Sharma et al., 2023) — shows human/preference-model data can reward sycophantic answers. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
48. **Rule-Based Rewards for Language Model Safety** (OpenAI, 2024) — explicit, rule-based reward signals used alongside RLHF for safety training. [OpenAI PDF](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf)
49. **Mitigating the Alignment Tax of RLHF** (Lin et al., 2023) — techniques to reduce capability regressions caused by safety tuning. [arXiv:2309.06256](https://arxiv.org/abs/2309.06256)
50. **Concrete Problems in AI Safety** (Amodei et al., 2016) — foundational taxonomy (reward hacking, side effects, safe exploration) motivating much RLHF safety work. [arXiv:1606.06565](https://arxiv.org/abs/1606.06565)
51. **Categorizing Variants of Goodhart's Law** (Manheim & Garrabrant, 2018) — formal taxonomy of reward-proxy failure relevant to reward hacking. [arXiv:1803.04585](https://arxiv.org/abs/1803.04585)
52. **Debating with More Persuasive LLMs Leads to More Truthful Answers** (Khan et al., 2024) — RL-trained persuasiveness improves debate-based oversight. [arXiv:2402.06782](https://arxiv.org/abs/2402.06782)
53. **Red Teaming Language Models to Reduce Harms** (Ganguli et al., 2022) — systematic adversarial probing used to harden RLHF-tuned models. [arXiv:2209.07858](https://arxiv.org/abs/2209.07858)

## Scalable Oversight & Supervising Superhuman Models (54–63)
54. **AI Safety via Debate** (Irving, Christiano & Amodei, 2018) — two agents debate so a human judge can supervise tasks beyond their own ability. [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
55. **Scalable AI Safety via Doubly-Efficient Debate** (Brown-Cohen et al., 2023) — bounded-compute honest-prover debate protocols. [arXiv:2311.14125](https://arxiv.org/abs/2311.14125)
56. **Debate Helps Supervise Unreliable Experts** (Michael et al., 2023) — empirical test of debate as a scalable-oversight method. [arXiv:2311.08702](https://arxiv.org/abs/2311.08702)
57. **Iterated Amplification** (Christiano, Shlegeris & Amodei, 2018) — supervising strong learners by amplifying weak human experts. [arXiv:1810.08575](https://arxiv.org/abs/1810.08575)
58. **Scalable Agent Alignment via Reward Modeling** (Leike et al., 2018) — DeepMind's research direction for recursive reward modeling. [arXiv:1811.07871](https://arxiv.org/abs/1811.07871)
59. **Weak-to-Strong Generalization** (Burns et al., 2023) — an analogy/testbed for humans supervising superhuman models. [arXiv:2312.09390](https://arxiv.org/abs/2312.09390)
60. **Discovering Agents** (Kenton et al., 2022) — formal criteria for detecting agency, relevant to oversight of learned systems. [arXiv:2208.08345](https://arxiv.org/abs/2208.08345)
61. **Cooperative Inverse Reinforcement Learning (CIRL)** (Hadfield-Menell et al., 2016) — human and AI jointly optimize a shared, initially uncertain reward. [arXiv:1606.03137](https://arxiv.org/abs/1606.03137)
62. **Multi-Principal Assistance Games** (Fickinger et al., 2020) — extends assistance games to multiple, possibly disagreeing, human principals. [arXiv:2007.09540](https://arxiv.org/abs/2007.09540)
63. **Quantifying the Gain in Weak-to-Strong Generalization** (Charikar et al., 2024) — theory bounding achievable weak-to-strong performance gains. [arXiv:2405.15116](https://arxiv.org/abs/2405.15116)

## Process Supervision & Reasoning-Focused RL (64–71)
64. **Let's Verify Step by Step** (Lightman et al., 2023) — process reward models supervise each reasoning step rather than only the final answer. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
65. **Training Verifiers to Solve Math Word Problems** (Cobbe et al., 2021) — introduces GSM8K and reward-model-guided answer verification. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
66. **STaR: Bootstrapping Reasoning with Reasoning** (Zelikman et al., 2022) — self-generated rationales become their own training signal. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
67. **Automated Process Supervision** (Luo et al., 2024) — trains process reward models without costly human step-level labels. [arXiv:2406.06592](https://arxiv.org/abs/2406.06592)
68. **Process Reinforcement through Implicit Rewards (PRIME)** (Liu et al., 2025) — dense, implicit process rewards for RL training. [arXiv:2502.01456](https://arxiv.org/abs/2502.01456)
69. **Scaling LLM Test-Time Compute Optimally** (Snell et al., 2024) — trades train-time RL for inference-time search and verification. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
70. **Inference-Time Scaling for Generalist Reward Modeling** (Liu et al., 2025) — scales reward-model compute at inference rather than training time. [arXiv:2504.02495](https://arxiv.org/abs/2504.02495)
71. **Skywork-Reward: Bag of Tricks for Reward Modeling** — practical recipes for training strong, robust reward models. [arXiv:2410.18451](https://arxiv.org/abs/2410.18451)

## Failure Modes, Diagnostics & Critical Theory (72–81)
72. **Open Problems and Fundamental Limitations of RLHF** (Casper et al., 2023) — comprehensive critique spanning the whole RLHF pipeline. [arXiv:2307.15217](https://arxiv.org/abs/2307.15217)
73. **AI Alignment: A Comprehensive Survey** (Ji et al., 2023) — broad survey spanning RLHF and adjacent alignment/safety methods. [arXiv:2310.19852](https://arxiv.org/abs/2310.19852)
74. **Foundational Challenges in Assuring Alignment and Safety of LLMs** (Anwar et al., 2024) — survey of open problems across the alignment stack. [arXiv:2404.09932](https://arxiv.org/abs/2404.09932)
75. **Reward Learning from Human Preferences and Demonstrations in Atari** (Ibarz et al., 2018) — combines preference comparisons with demonstrations for reward learning. [arXiv:1811.06521](https://arxiv.org/abs/1811.06521)
76. **Weak-to-Strong Preference Optimization** (Zhu et al., 2024) — "steals" a usable reward signal from a weak, already-aligned model. [arXiv:2410.18640](https://arxiv.org/abs/2410.18640)
77. **Contrastive Weak-to-Strong Generalization** — contrastive training objective to improve weak-to-strong transfer. [arXiv:2510.07884](https://arxiv.org/abs/2510.07884)
78. **Reward Model Learning vs. Direct Policy Optimization** — comparative analysis of the two main RLHF paradigms. [arXiv:2403.01857](https://arxiv.org/abs/2403.01857)
79. **The Energy Loss Phenomenon in RLHF** — a new diagnostic lens on reward hacking during PPO-style training. [arXiv:2501.19358](https://arxiv.org/abs/2501.19358)
80. **Segmenting Text and Learning Rewards for Improved RLHF** — finer-grained credit assignment for reward models. [arXiv:2501.02790](https://arxiv.org/abs/2501.02790)
81. **Adaptive Margin RLHF via Preference over Preferences** — uses meta-preferences to set adaptive margins in DPO-style training. [arXiv:2509.22851](https://arxiv.org/abs/2509.22851)

## Evaluation, Benchmarks, Datasets & Tooling (82–90)
82. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., 2023) — the LLM-judge evaluation methodology underlying many RLHF pipelines. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
83. **Length-Controlled AlpacaEval** (Dubois et al., 2024) — debiases automatic win-rate evaluators used to score RLHF/DPO models. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)
84. **AlpacaFarm** (Dubois et al., 2023) — low-cost simulation sandbox for comparing RLHF methods. [arXiv:2305.14387](https://arxiv.org/abs/2305.14387)
85. **HelpSteer3** (Wang et al., 2025) — open, human-annotated, multi-task, multilingual preference dataset. [arXiv:2505.11475](https://arxiv.org/abs/2505.11475)
86. **TRL (Transformer Reinforcement Learning)** — open-source library implementing PPO, DPO, KTO, and other RLHF trainers. [GitHub](https://github.com/huggingface/trl)
87. **Safe-RLHF / PKU-SafeRLHF** — open codebase, Beaver models, and safety-labeled preference dataset. [GitHub](https://github.com/PKU-Alignment/safe-rlhf)
88. **Skywork-Reward-V2** — scales preference-data curation via human-AI synergy for reward-model training. [arXiv:2507.01352](https://arxiv.org/abs/2507.01352)
89. **Awesome-RLHF-Vision** — curated RLHF resources specific to vision and multimodal models. [GitHub](https://github.com/chengjl19/Awesome-RLHF-Vision)
90. **Optimal Design for Human Preference Elicitation** — experiment-design methods to reduce the cost of preference labeling. [arXiv:2404.13895](https://arxiv.org/abs/2404.13895)

## Constrained & Bias/Safety-Theoretic Extensions of DPO/RLHF (91–100)
91. **Direct Preference Optimization with an Offset** — margin-based refinement of the standard DPO loss. [arXiv:2402.10571](https://arxiv.org/abs/2402.10571)
92. **BiasDPO** — uses DPO-style training specifically to reduce social bias in language-model outputs. [arXiv:2407.13928](https://arxiv.org/abs/2407.13928)
93. **Filtered Direct Preference Optimization** — filters noisy or low-quality preference pairs before DPO training. [arXiv:2404.13846](https://arxiv.org/abs/2404.13846)
94. **LOGO: Long-Context Alignment via Efficient Preference Optimization** — extends preference optimization to long-context settings. [arXiv:2410.18533](https://arxiv.org/abs/2410.18533)
95. **Provably Convergent Primal-Dual DPO** — safety-constrained DPO with formal convergence guarantees. [arXiv:2510.05703](https://arxiv.org/abs/2510.05703)
96. **Certifiable Safe RLHF** — semantic grounding plus fixed-penalty constrained optimization for safer alignment. [arXiv:2510.03520](https://arxiv.org/abs/2510.03520)
97. **Learning Natural Language Constraints for Safe RL of Language Agents** — derives explicit safety constraints for constrained RL fine-tuning. [arXiv:2504.03185](https://arxiv.org/abs/2504.03185)
98. **Safety Alignment via Contrasting Safe and Harmful Distributions** — contrastive training signal for harmlessness alignment. [arXiv:2406.16743](https://arxiv.org/abs/2406.16743)
99. **Vulnerabilities Mitigation for Safety-Aligned LMs via Debiasing** — addresses limits of single-objective safe-RLHF/DPO variants across domains. [arXiv:2502.02153](https://arxiv.org/abs/2502.02153)
100. **RLHF Deciphered: A Critical Analysis of RLHF for LLMs** — synthesis and critique tying together the methods above. [arXiv:2404.08555](https://arxiv.org/abs/2404.08555)

---
*Compiled from arXiv, official model-provider publications, and maintained GitHub resource lists. Most entries are individual papers introducing or analyzing a specific method; a few (marked "Awesome-*", "TRL", dataset repos) are maintained resource collections rather than single papers.*





//////////////////////////////////////////////////////////////////////////////////////////////////////////
rough draft to fix the math notation presented by the ai models 

> [!abstract] Unified Framework: Policy Optimization from Feedback (Detailed)
> 
> This framework unifies Supervised Fine-Tuning (SFT), Preference Fine-Tuning, and Reinforcement Learning with Verifiable Rewards (RLVR) as special cases of a single optimization problem.
> 
> **Symbols and their origins:**
> - $x \in \mathcal{X}$: a prompt or input. Drawn from a prompt distribution $\mathcal{D}$.
> - $y \in \mathcal{Y}$: a response or completion. Can be a sequence of tokens $y=(y_1,\dots,y_T)$.
> - $\pi_\theta(y\mid x)$: the policy, a parametric model (e.g., a transformer) that outputs a probability distribution over responses given a prompt. $\theta$ are the parameters.
> - $\mathcal{D}$: the distribution over prompts. Often a fixed dataset of prompts, but can be a broader distribution.
> - $f$: a feedback signal. Its nature differs per method: a demonstration, a preference pair, or a verifier score.
> - $\mu_\theta(y\mid x)$: the distribution from which responses are sampled during training. It can be the data distribution (off-policy) or the policy itself (on-policy).
> - $\ell(\theta; x, y)$: a loss function measuring how poorly the policy matches the feedback for a given prompt and response.
> - $\Omega(\theta)$: a regularizer, typically a KL divergence to a reference policy $\pi_{\text{ref}}$. It prevents the policy from drifting too far, preserving fluency and preventing reward hacking.
> - $\beta \ge 0$: a scalar controlling the strength of the regularizer.
> 
> **General objective:**
> $$
> \theta^*
> =
> \arg\min_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D}}
> \;
> \mathbb{E}_{y\sim \mu_\theta(\cdot\mid x)}
> \left[
> \ell(\theta; x, y)
> \right]
> +
> \beta\,\Omega(\theta)
> $$
> 
> The key differences between methods are:
> 1. What feedback $f$ is used.
> 2. Whether responses $y$ are sampled from data or from the policy.
> 3. What loss $\ell$ is applied.
> 4. Whether a regularizer $\Omega$ is used.
> 
> All three methods below are instances of this template.

> [!note] 1. Supervised Fine-Tuning (SFT)
> 
> **Feedback:** a demonstration $f = y$. That is, for each prompt $x$, we have a human-written (or otherwise high-quality) response $y$.
> 
> **Response source:** data distribution. We sample $y$ from the dataset, not from the policy:
> $$
> \mu_\theta(y\mid x) = \mathcal{D}_{\text{SFT}}(y\mid x)
> $$
> Here $\mathcal{D}_{\text{SFT}}$ is the empirical distribution of the supervised fine-tuning dataset, typically pairs $(x,y)$.
> 
> **Loss:** negative log-likelihood (NLL). This is the standard maximum likelihood estimation (MLE) loss:
> $$
> \ell_{\text{SFT}}(\theta; x, y) = -\log \pi_\theta(y\mid x)
> $$
> For token sequences, this expands to:
> $$
> -\log \pi_\theta(y\mid x) = -\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})
> $$
> 
> **Why this loss?** Minimizing NLL maximizes the probability of the demonstrated responses. It is equivalent to minimizing the forward KL divergence between the data distribution and the policy.
> 
> **Objective:**
> $$
> \theta_{\text{SFT}}
> =
> \arg\min_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D}}
> \;
> \mathbb{E}_{y\sim \mathcal{D}_{\text{SFT}}(\cdot\mid x)}
> \left[
> -\log \pi_\theta(y\mid x)
> \right]
> $$
> 
> **Regularizer:** typically none. Sometimes a small weight decay or dropout is used, but no explicit KL term to a reference policy.
> 
> **Symbol origins:** $\mathcal{D}_{\text{SFT}}$ is the supervised dataset, often collected from human experts or strong models. $x$ and $y$ are the prompt and response from that dataset.

> [!tip] 2. Preference Fine-Tuning
> 
> **Feedback:** pairwise preference $f = (y^+, y^-)$, where $y^+$ is preferred over $y^-$ for prompt $x$. This is a comparative signal, not an absolute demonstration.
> 
> **Response source:** data distribution over pairs. We sample preference pairs from a dataset:
> $$
> \mu_\theta(y^+,y^-\mid x) = \mathcal{D}_{\text{pref}}(y^+,y^-\mid x)
> $$
> $\mathcal{D}_{\text{pref}}$ is the empirical distribution of the preference dataset.
> 
> **Score function:** let $s_\theta(x,y)$ be any score derived from the policy. Common choices:
> - Log-probability: $s_\theta(x,y) = \log \pi_\theta(y\mid x)$.
> - Implicit reward (as in DPO): $s_\theta(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$.
> - Explicit reward model: $s_\theta(x,y) = r_\phi(x,y)$, a separately learned reward model.
> 
> **Loss:** a general ranking loss. The most common is the logistic loss based on the Bradley–Terry model. The Bradley–Terry model assumes:
> $$
> P(y^+ \succ y^- \mid x) = \sigma\bigl(s_\theta(x,y^+) - s_\theta(x,y^-)\bigr)
> $$
> where $\sigma$ is the logistic sigmoid. The negative log-likelihood of this model gives:
> $$
> \ell_{\text{pref}}(\theta; x, y^+, y^-)
> =
> -\log \sigma\bigl(s_\theta(x,y^+) - s_\theta(x,y^-)\bigr)
> $$
> More generally, any monotone pairwise loss $\ell(s_\theta(y^+), s_\theta(y^-))$ that penalizes $s_\theta(y^+) \le s_\theta(y^-)$ can be used (e.g., hinge loss, margin loss).
> 
> **Objective:**
> $$
> \theta_{\text{pref}}
> =
> \arg\min_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D}}
> \;
> \mathbb{E}_{(y^+,y^-)\sim \mathcal{D}_{\text{pref}}(\cdot\mid x)}
> \left[
> \ell\bigl(s_\theta(x,y^+), s_\theta(x,y^-)\bigr)
> \right]
> +
> \beta\,\Omega(\theta)
> $$
> 
> **Regularizer:** often a KL divergence to a reference policy $\pi_{\text{ref}}$:
> $$
> \Omega(\theta) = \mathbb{E}_{x\sim\mathcal{D}} \left[ D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x) \,\|\, \pi_{\text{ref}}(\cdot\mid x)\bigr) \right]
> $$
> In methods like DPO, the KL regularization is implicit in the choice of the implicit reward $s_\theta$.
> 
> **How it relates to RLHF:** In classic RLHF, one first learns a reward model $r_\phi(x,y)$ from preferences using the Bradley–Terry loss, then optimizes the policy with RL to maximize $r_\phi$ minus a KL penalty. Here, preference fine-tuning directly optimizes the policy on the preference pairs, bypassing the explicit reward model (as in DPO) or using an implicit reward.
> 
> **Symbol origins:** $y^+$ and $y^-$ are the preferred and dispreferred responses, typically generated by models and annotated by humans. $\mathcal{D}_{\text{pref}}$ is the resulting dataset.

> [!warning] 3. RLVR: Reinforcement Learning with Verifiable Rewards
> 
> **Feedback:** a verifier $f = V(x,\cdot)$, where $V(x,y)\in\mathbb{R}$ is a programmatically checkable reward. Unlike human preferences, $V$ is deterministic and automatic. Examples:
> - Exact match to a ground-truth answer.
> - Unit tests passing for code.
> - Formal proof checking.
> - Schema or format validation.
> 
> **Response source:** the policy itself. We generate responses on-policy:
> $$
> \mu_\theta(y\mid x) = \pi_\theta(y\mid x)
> $$
> This means the policy explores its own outputs and receives verifier scores.
> 
> **Loss:** negative verifier reward:
> $$
> \ell_{\text{RLVR}}(\theta; x, y) = -V(x,y)
> $$
> 
> **Objective (min form):**
> $$
> \theta_{\text{RLVR}}
> =
> \arg\min_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D}}
> \;
> \mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
> \left[
> -V(x,y)
> \right]
> +
> \beta\,
> \mathbb{E}_{x\sim\mathcal{D}}
> \left[
> D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\bigr)
> \right]
> $$
> 
> **Objective (max form):**
> $$
> \theta_{\text{RLVR}}
> =
> \arg\max_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D},\, y\sim \pi_\theta(\cdot\mid x)}
> \bigl[V(x,y)\bigr]
> -
> \beta\,
> \mathbb{E}_{x\sim\mathcal{D}}
> \left[
> D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\bigr)
> \right]
> $$
> 
> **Why the KL regularizer?** Without it, the policy may drift to degenerate solutions that exploit the verifier (reward hacking) or lose fluency. The reference policy $\pi_{\text{ref}}$ is typically the SFT model.
> 
> **Algorithms:** This objective is usually optimized with policy gradient methods like PPO or GRPO. In PPO, one uses a clipped surrogate objective with advantages computed from $V$. In GRPO, advantages are computed relative to a group of sampled responses for the same prompt.
> 
> **Symbol origins:** $V$ is a verifier function, often hand-coded or based on a ground-truth database. $\mathcal{D}$ is a set of prompts (no responses needed). $\pi_{\text{ref}}$ is a frozen reference policy, usually the SFT checkpoint.

> [!summary] Unified Comparison
> 
> | Method | Feedback \(f\) | Response source | Loss \(\ell\) | Regularizer \(\Omega\) | Typical algorithms |
> |---|---|---|---|---|---|
> | **SFT** | Demonstration \(y\) | Data \(\mathcal{D}_{\text{SFT}}\) | \(-\log \pi_\theta(y\mid x)\) | None | Gradient descent on NLL |
> | **Preference FT** | Pair \((y^+,y^-)\) | Data \(\mathcal{D}_{\text{pref}}\) | Ranking loss \(\ell(s_\theta(y^+),s_\theta(y^-))\) | Optional KL to \(\pi_{\text{ref}}\) | DPO, IPO, KTO, RLHF (with RM) |
> | **RLVR** | Verifier \(V(x,y)\) | Policy \(\pi_\theta\) | \(-V(x,y)\) | KL to \(\pi_{\text{ref}}\) | PPO, GRPO, REINFORCE |
> 
> **Unified form:**
> $$
> \theta^*
> =
> \arg\min_\theta
> \;
> \mathbb{E}_{x\sim\mathcal{D}}
> \;
> \mathbb{E}_{y\sim \mu_\theta(\cdot\mid x)}
> \left[
> \ell(\theta; x, y)
> \right]
> +
> \beta\,\Omega(\theta)
> $$
> 
> In words:
> - **SFT** uses off-policy demonstrations and a likelihood loss.
> - **Preference fine-tuning** uses off-policy pairwise comparisons and a ranking loss.
> - **RLVR** uses on-policy generation and a verifiable reward.
> 
> All three are instances of the same general method: optimize a policy against a feedback signal, with an optional regularizer to keep the policy close to a reference.

# AI Safety Methods Built on RLHF, RLVR & Supervised Fine-Tuning

A curated reference of published, verifiable techniques whose training mechanism is **RLHF** (Reinforcement Learning from Human Feedback), **RLVR** (Reinforcement Learning with Verifiable Rewards), or **SFT** (Supervised Fine-Tuning), applied specifically to AI safety objectives (harmlessness, honesty, robustness, oversight, alignment). Every entry links to a real primary source I located and checked.

**A note on scope, since 100 fully independent methods don't really exist in the literature:** I found roughly 90 genuinely distinct, real, documented methods/papers. Rather than pad the list to a round 100 with invented or tangential entries, I stopped at what's real and grouped closely-related variants (e.g. the DPO family) as clearly-labeled sub-entries. Section 9 (Preference-Optimization Family) is flagged separately because those methods are technically **RL-free** — they replace RLHF's RL step with a closed-form loss — but they're used interchangeably with RLHF inside the same SFT→preference-alignment safety pipelines, so I've included them with that caveat rather than silently miscategorizing them as RLHF.

---

## 1. Core RLHF-for-Safety Methods

1. **RLHF (Deep RL from Human Preferences)** — Christiano et al., 2017. The foundational algorithm: a reward model is trained from pairwise human comparisons and a policy is optimized against it via RL. Every later safety-RLHF method builds on this. https://arxiv.org/abs/1706.03741
2. **InstructGPT** — Ouyang et al., 2022 (OpenAI). SFT on human demonstrations, then RLHF (PPO) against a reward model, explicitly to reduce toxic, untruthful and unhelpful outputs. https://arxiv.org/abs/2203.02155
3. **Learning to Summarize from Human Feedback** — Stiennon et al., 2020 (OpenAI). One of the first large-scale SFT+reward-model+PPO pipelines, establishing the recipe later safety-RLHF work reuses. https://arxiv.org/abs/2009.01325
4. **WebGPT** — Nakano et al., 2021 (OpenAI). SFT via behavior cloning plus RLHF/rejection sampling, requiring cited web evidence to cut down false and harmful claims. https://arxiv.org/abs/2112.09332
5. **HH-RLHF: Training a Helpful and Harmless Assistant** — Bai et al., 2022 (Anthropic). Applies RLHF explicitly to jointly optimize helpfulness *and* harmlessness, and studies how hard RLHF models are to red-team as they scale. https://arxiv.org/abs/2204.05862
6. **A General Language Assistant as a Laboratory for Alignment** — Askell et al., 2021 (Anthropic). Introduces the "Helpful, Honest, Harmless" (HHH) framework, comparing SFT/context-distillation and preference-model RL as safety-alignment tools. https://arxiv.org/abs/2112.00861
7. **Red Teaming Language Models to Reduce Harms** — Ganguli et al., 2022 (Anthropic). Uses large-scale adversarial human red-teaming to train and evaluate RLHF models, releasing 38,961 attack transcripts. https://arxiv.org/abs/2209.07858
8. **Sparrow** — Glaese et al., 2022 (DeepMind). RLHF with a dedicated *rule-violation* reward model (23 hand-written safety rules) trained alongside a preference reward model. https://arxiv.org/abs/2209.14375
9. **Safe RLHF** — Dai et al., 2023. Decouples helpfulness (reward model) from harmlessness (cost model) and solves a Lagrangian-constrained RLHF objective so harm stays under a cap. https://arxiv.org/abs/2310.12773
10. **Safe RLHF-V** — Ji et al., 2025. Extends Safe RLHF's dual reward/cost Lagrangian approach to multimodal LLMs, cutting hallucinated/harmful vision-language outputs. https://arxiv.org/abs/2503.17682
11. **HC-RLHF (High-Confidence Safety Constraints)** — Chittepu et al., 2025. Adds a Seldonian high-confidence statistical safety guarantee on top of RLHF so the trained policy provably won't exceed a harm tolerance. https://arxiv.org/abs/2506.08266
12. **Certifiable Safe RLHF** — 2025. Adds semantic grounding and a fixed-penalty constrained-optimization scheme to Safe RLHF's cost model for more reliable jailbreak resistance. https://arxiv.org/abs/2510.03520
13. **PKU-SafeRLHF (multi-level safety dataset & training)** — Ji et al., 2024. Multi-level (minor/moderate/severe) human-preference safety data used to iteratively RLHF-train and red-team LLMs. https://arxiv.org/abs/2406.15513
14. **Rule-Based Rewards (RBR) for Safety** — Mu, Helyar et al., 2024 (OpenAI). Feeds LLM-graded, rule-based safety propositions directly into the RLHF reward alongside the human-trained reward model. https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf
15. **Scaling Laws for Reward Model Overoptimization** — Gao, Schulman & Hilton, 2022 (OpenAI). Empirically measures Goodhart's-Law reward hacking under RLHF/best-of-N so teams can bound how far to optimize safely. https://arxiv.org/abs/2210.10760
16. **BeaverTails-driven Safety RLHF** — Ji et al., 2023. A 330K-pair human-preference safety dataset (separating helpfulness/harmlessness labels) purpose-built to drive RLHF safety training. https://arxiv.org/abs/2307.04657

## 2. Constitutional AI & RLAIF Family

17. **Constitutional AI (CAI / RLAIF)** — Bai et al., 2022 (Anthropic). SFT critique-and-revise stage, then RL from *AI* (not human) feedback scored against a written constitution instead of crowd-worker labels. https://arxiv.org/abs/2212.08073
18. **Specific versus General Principles for Constitutional AI** — Kundu et al., 2023 (Anthropic). Tests whether one general CAI principle ("do what's best for humanity") suffices to suppress power-seeking/self-preservation statements. https://arxiv.org/abs/2310.13798
19. **Collective Constitutional AI** — Anthropic & Collective Intelligence Project, 2023. Trains a CAI model against a constitution drafted through public deliberation rather than an in-house one. https://dl.acm.org/doi/10.1145/3630106.3658979
20. **Multi-Objective RLAIF (MORLAIF)** — 2024. Trains separate AI-feedback preference models per moral principle instead of one merged constitution, for finer-grained harmlessness control. https://arxiv.org/abs/2406.07295
21. **IterAlign** — 2024. Automatically mines new constitutional principles from red-team failures and iterates SFT+RLAIF rounds to patch them. https://arxiv.org/abs/2403.18341
22. **AutoRule** — 2025. Extracts rules automatically from chain-of-thought critiques and converts them into rule-based RL rewards, extending the Constitutional AI/Sparrow lineage. https://arxiv.org/abs/2506.15651
23. **Constitutional Classifiers** — Sharma et al., 2025 (Anthropic). Input/output safety classifiers SFT-trained on constitution-derived synthetic data to block universal jailbreaks with minimal over-refusal. https://arxiv.org/abs/2501.18837

## 3. Scalable Oversight, Debate & Amplification

24. **AI Safety via Debate** — Irving, Christiano & Amodei, 2018 (OpenAI). Two AIs argue opposing answers; a human/weaker judge's verdict becomes the RL reward signal, rewarding the more truthful debater. https://arxiv.org/abs/1805.00899
25. **Scalable AI Safety via Doubly-Efficient Debate** — Brown-Cohen, Irving & Piliouras, 2023 (DeepMind). Gives debate protocols where the honest strategy needs only polynomial-time simulation, making RL-trained debate judges practical. https://arxiv.org/abs/2311.14125
26. **An Alignment Safety Case Sketch Based on Debate** — 2025. Lays out what further RL-based debate training and evidence would be needed to trust debate as a genuine safety argument. https://arxiv.org/abs/2505.03989
27. **Debating with More Persuasive LLMs Leads to More Truthful Answers** — Khan et al., 2024. Empirically tests debate as scalable oversight by training/using stronger debaters judged by weaker ones. https://arxiv.org/abs/2402.06782
28. **Self-Critiquing Models for Assisting Human Evaluators** — Saunders et al., 2022 (OpenAI). SFTs (behavioral cloning) a model to write natural-language critiques so humans can supervise outputs too complex to check unaided. https://arxiv.org/abs/2206.05802
29. **Recursive Reward Modeling** — Leike et al., 2018 (DeepMind). Uses agents already trained via reward modeling to assist the human evaluator in supervising the next, more capable agent. https://arxiv.org/abs/1811.07871
30. **Iterated Distillation and Amplification (IDA)** — Christiano et al., 2018. Alternates "amplifying" a human's judgment with many AI-assisted calls and "distilling" the amplified judgment back into a policy via supervised learning. https://www.alignmentforum.org/s/EmDuGeRw749sD3GKd
31. **Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning** — Sang et al., 2024. Combines weak-to-strong RL supervision with AI-debate-based oversight and ensembled weak teachers. https://arxiv.org/abs/2402.00667

## 4. Weak-to-Strong / Superalignment

32. **Weak-to-Strong Generalization** — Burns et al., 2023 (OpenAI Superalignment). Fine-tunes a strong pretrained model on labels from a much weaker supervisor, as a testable analogy for humans overseeing superhuman AI. https://openai.com/index/weak-to-strong-generalization/
33. **Selective Weak-to-Strong Generalization** — 2025. Lets the strong model abstain from imitating weak labels it's confident are wrong, reducing propagation of the weak supervisor's safety-relevant mistakes. https://arxiv.org/abs/2511.14166
34. **Aligner: Efficient Alignment through Weak-to-Strong Correction** — Ji et al., 2024. Trains a small SFT "correction" model on query-answer-correction triples that plugs onto any upstream model to boost harmlessness without needing its weights. https://arxiv.org/abs/2402.02416
35. **ConTrans: Weak-to-Strong Alignment Engineering via Concept Transplantation** — 2024. Transplants safety-relevant concept vectors from a small RLHF-aligned model into a larger unaligned model's residual stream. https://arxiv.org/abs/2405.13578

## 5. RLVR-Based Safety Methods

36. **RLVR for Safety Reasoning — AlphaAlign** — 2025. Treats refusal-worthiness as a verifiable, binary reward signal so a model develops safety chain-of-thought reasoning without any hand-written safety CoT data. https://arxiv.org/abs/2507.14987
37. **Breaking the Safety-Capability Tradeoff (RLVR maintains guardrails)** — 2025. Shows theoretically and empirically that KL-regularized RLVR (unlike SFT or vanilla RLHF) can boost math/code capability while leaving safety metrics essentially unchanged. https://arxiv.org/abs/2511.21050
38. **Deliberative Alignment** — Guan et al., 2024 (OpenAI). SFT on (prompt, CoT, output) triples that reference safety-policy text, plus an RLVR-style RL stage judged against those policies; used to align OpenAI's o-series models. https://arxiv.org/abs/2412.16339
39. **IFDecorator (RLVR reward-hacking mitigation)** — 2025. Wraps RLVR training with an adversarial data flywheel, an intent-alignment checker, and "trip wire" traps that catch reward-hacking/shortcut exploitation. https://arxiv.org/abs/2508.04632
40. **Case-Augmented Deliberative Alignment** — 2026. Compares training on explicit safety codes vs. illustrative cases within the deliberative-alignment RLVR/SFT recipe, finding case-based training generalizes safety behavior better. https://arxiv.org/abs/2601.08000
41. **HarmRLVR (adversarial red-team probe of RLVR)** — 2025. Not a defense itself, but the key finding safety-RLVR designers must contend with: safety alignment can be reversed with GRPO using only 64 harmful prompts. Included here as the robustness benchmark this whole subfield is now built against. https://arxiv.org/abs/2510.15499

## 6. Process & Fine-Grained Reward Methods

42. **Let's Verify Step by Step** — Lightman et al., 2023 (OpenAI). Introduces process reward models (PRMs) that grade each reasoning step rather than only the final answer, released with the 800K-label PRM800K dataset. https://arxiv.org/abs/2305.20050
43. **Fine-Grained RLHF** — Wu et al., 2023. Rewards sentence/sub-sentence spans for specific error types (false, toxic, irrelevant) instead of one holistic score, cutting toxicity and improving sample efficiency. https://arxiv.org/abs/2306.01693
44. **RLHF-V** — Yu et al., 2023. Collects segment-level human corrections of multimodal hallucinations and performs dense preference optimization over them, cutting hallucination rate by ~35%. https://arxiv.org/abs/2312.00849
45. **Math-Shepherd** — 2023. Automatically builds step-level PRM training data without human annotation, extending process supervision's reach for verifiable-reasoning safety/quality checks. https://arxiv.org/abs/2312.08935
46. **TLCR (Token-Level Continuous Reward)** — 2024. Trains a discriminator to assign continuous, token-level rewards for RLHF, aimed at reducing risks from coarse, sequence-level reward signals. https://arxiv.org/abs/2407.16574
47. **Sparrow's Rule Reward Model** — Glaese et al., 2022 (DeepMind).
(Cross-referenced from §1/§2.) A dedicated PRM-like classifier estimating
per-turn rule-violation probability, trained on 14K+ human-annotated
conversations. https://arxiv.org/abs/2209.14375

## 7. Supervised Fine-Tuning Safety Methods
****
48. **Safety-Tuned LLaMAs** — Bianchi et al., 2023. Shows that adding just ~3% safety refusal examples to SFT data measurably reduces harmful compliance without a full RLHF pass. https://arxiv.org/abs/2309.07875
49. **SFT-Initialized Red-Teaming Agents** — Guo et al., 2025 / Perez et al. Uses SFT on human or self-play red-team dialogues to bootstrap an "attacker" model so it doesn't just refuse to red-team, later refined with RL. https://arxiv.org/abs/2510.02286
50. **Holistic Automated Red Teaming via SFT Cloning** — 2024. SFTs a red-team agent on Anthropic's public red-team transcripts by masking assistant turns and fitting to the human attacker's turns instead. https://arxiv.org/abs/2409.16783
51. **IntentionReasoner** — 2025. SFT "cold start" on ~163K deduplicated red-team + benign queries to teach a model to reason about user intent before deciding whether/how to refuse. https://arxiv.org/abs/2508.20151
52. **Magic-Token-Guided Safety Co-Training** — 2025. A single SFT stage that jointly trains positive/negative/refusal safety "modes," switchable at inference via a system-level token, matching SFT+DPO safety quality. https://arxiv.org/abs/2508.14904
53. **Think Before Refusal** — 2025. SFT on a small, curated instruction set (general + labeled safety queries) to trigger a "safety reflection" step, reducing false refusals of benign prompts. https://arxiv.org/abs/2503.17882
54. **SafeDecoding** — 2024. Fine-tunes a small "expert" SFT model on red-team refusal data, then blends its token distribution with the base model's at decoding time to defend against jailbreaks. https://arxiv.org/abs/2402.08983
55. **Proactive Safety Reasoning via SFT** — 2025. SFTs a model on a harmfulness-assessment chain-of-thought plus a benign "retain set," so the model reasons about risk before answering rather than pattern-matching refusals. https://arxiv.org/abs/2501.19180
56. **Context Distillation for HHH** — Askell et al., 2021 (Anthropic). Distills a long "helpful, honest, harmless" prompt into model weights via SFT, so the safety framing survives without needing the prompt at inference. https://arxiv.org/abs/2112.00861
57. **Constitutional AI's SFT (Critique-Revision) Stage** — Bai et al., 2022 (Anthropic). (Cross-referenced from §2.) The supervised half of CAI: self-critique-and-revise transcripts are used to SFT a harmless base model before the RLAIF stage begins. https://arxiv.org/abs/2212.08073
58. **Deliberative Alignment's SFT Stage** — Guan et al., 2024 (OpenAI). (Cross-referenced from §5.) Teaches a model to recall and quote safety-policy text inside its own chain-of-thought via supervised fine-tuning on synthetic (prompt, CoT, answer) triples. https://arxiv.org/abs/2412.16339
59. **Constitutional Classifiers' SFT Training** — Sharma et al., 2025 (Anthropic). (Cross-referenced from §2.) The classifier guard itself is a model SFT-trained on constitution-conditioned synthetic harmful/benign examples, not RL-trained. https://arxiv.org/abs/2501.18837

## 8. Truthfulness, Hallucination & Sycophancy Mitigation

60. **Towards Understanding Sycophancy in Language Models** — Sharma et al., 2023 (Anthropic). Shows RLHF preference models can reward agreeable-but-false answers over truthful ones, motivating scalable-oversight fixes. https://arxiv.org/abs/2310.13548
61. **How RLHF Amplifies Sycophancy** — Shapira, Benade & Procaccia, 2026. A formal mechanistic account of the effect above, plus a principled reward-shaping correction to the RLHF objective. https://arxiv.org/abs/2602.01002
62. **On-Policy Self-Alignment with Fine-Grained Knowledge Feedback** — 2024. Uses RLHF-style on-policy optimization against a knowledge-grounded, fine-grained reward specifically to cut hallucination rates. https://arxiv.org/abs/2406.12221
63. **SMART: Sycophancy Mitigation via Uncertainty-Aware RL** — 2025. RL-trains models toward uncertainty-aware reasoning trajectories so they push back on incorrect user beliefs instead of conforming. https://arxiv.org/abs/2509.16742
64. **WebGPT's Citation-Grounded RLHF** — Nakano et al., 2021 (OpenAI). (Cross-referenced from §1.) Rewards answers backed by retrieved web citations specifically to improve measured truthfulness (TruthfulQA). https://arxiv.org/abs/2112.09332

## 9. Preference-Optimization Family (RL-free alternatives used inside RLHF-style safety pipelines)

> These replace RLHF's PPO step with a closed-form loss on preference pairs — no reward model, no on-policy RL — but they're routinely swapped into the *same* SFT → preference-alignment pipeline that RLHF and Constitutional AI use, including for safety objectives specifically (e.g. DPO was used in place of PPO in a CAI replication). Flagged here rather than folded silently into §1.

65. **DPO (Direct Preference Optimization)** — Rafailov et al., 2023. Reformulates the RLHF objective as a single classification loss on preference pairs, no reward model or RL loop needed. https://arxiv.org/abs/2305.18290
66. **IPO (Identity Preference Optimization)** — Azar et al., 2024. A general theoretical objective for learning from pairwise preferences that avoids DPO's implicit reward-overfitting assumption. https://arxiv.org/abs/2310.12036
67. **KTO (Kahneman-Tversky Optimization)** — Ethayarajh et al., 2024. Learns from unpaired binary "good/bad" feedback instead of preference pairs, using prospect-theoretic human-utility modeling. https://arxiv.org/abs/2402.01306
68. **ORPO (Odds-Ratio Preference Optimization)** — Hong et al., 2024. Drops the reference model entirely, folding a contrastive odds-ratio term directly into the SFT loss. https://arxiv.org/abs/2403.07691
69. **CPO (Contrastive Preference Optimization)** — Xu et al., 2024. Jointly optimizes sequence likelihood and a contrastive reward, merging SFT and alignment into one training pass. https://arxiv.org/abs/2401.08417
70. **RAFT (Reward-rAnked FineTuning)** — Dong et al., 2023. Samples a batch, scores it with a reward model, filters to the best-scoring subset, then SFTs on that filtered subset, iteratively. https://arxiv.org/abs/2304.06767
71. **ReST (Reinforced Self-Training)** — Gulcehre et al., 2023 (DeepMind). Alternates a "Grow" step (sample from the policy) with an "Improve" step (filter/rank by a reward model, then offline-RL fine-tune). https://arxiv.org/abs/2308.08998
72. **Constitution or Collapse? (CAI with DPO instead of PPO)** — 2025. A direct replication showing Constitutional AI's harmlessness gains hold when the RL stage is swapped for DPO on a smaller open model. https://arxiv.org/html/2504.04918v1

## 10. Robustness, Jailbreak Defense & Unlearning

73. **Circuit Breakers** — Zou et al., 2024. Fine-tunes a model to "short-circuit" internal representations associated with harmful generation, interrupting harmful completions mid-generation regardless of the jailbreak used. https://arxiv.org/abs/2406.04313
74. **Representation Engineering** — Zou et al., 2023. A top-down approach to reading and steering safety-relevant internal representations (e.g. honesty, harm) that underlies later methods like Circuit Breakers. https://arxiv.org/abs/2310.01405
75. **Latent Adversarial Training for Unforeseen Failure Modes** — Casper et al., 2024. Perturbs a model's latent activations adversarially during fine-tuning so it's robust to attacks not seen during training, unlike input-only adversarial training. https://arxiv.org/abs/2403.05030
76. **Many-Shot Jailbreaking (and its mitigation)** — Anil et al., 2024 (Anthropic). Identifies a long-context jailbreak technique and studies fine-tuning-based mitigations, which fed directly into Constitutional Classifiers. https://www.anthropic.com/research/many-shot-jailbreaking
77. **WMDP Benchmark & Unlearning** — Li et al., 2024. A benchmark plus an unlearning method (RMU) to excise hazardous bio/cyber/chem knowledge from model weights while preserving general capability. https://arxiv.org/abs/2403.03218
78. **Aligner (weak-to-strong SFT correction)** — Ji et al., 2024. (Cross-referenced from §4.) Its harmlessness gains are specifically validated against jailbreak-style adversarial prompts across 11 base models. https://arxiv.org/abs/2402.02416
79. **Constitutional Classifiers++** — 2026 (Anthropic). Production-efficiency successor to Constitutional Classifiers, re-validated against newer jailbreak families (e.g. Boundary Point Jailbreaking). https://arxiv.org/html/2601.04603v1
80. **SafeDecoding** — 2024. (Cross-referenced from §7.) An SFT-trained "safety expert" model whose token distribution is blended in at decode time specifically to blunt jailbreak prompts. https://arxiv.org/abs/2402.08983

## 11. Rule-Based, Cost-Constrained & Multi-Objective RLHF Variants

81. **Sparrow's Rule-Conditioned RM** — Glaese et al., 2022. (Cross-referenced.) 23 natural-language rules, each with its own violation classifier, combined additively into the RL reward. https://arxiv.org/abs/2209.14375
82. **Rule-Based Rewards (RBR)** — Mu et al., 2024. (Cross-referenced from §1.) Differs from Sparrow by using AI feedback instead of ~14K human-annotated rule-violation labels. https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf
83. **Safe RLHF's Lagrangian Constraint Method** — Dai et al., 2023. (Cross-referenced from §1.) The specific mathematical mechanism — a dynamically-tuned Lagrange multiplier — that trades off the reward and cost models during PPO. https://arxiv.org/abs/2310.12773
84. **HC-RLHF's Seldonian Safety Guarantee** — Chittepu et al., 2025. (Cross-referenced from §1.) Replaces Safe RLHF's expectation-based constraint with a high-confidence probabilistic bound. https://arxiv.org/abs/2506.08266
85. **PKU-SafeRLHF's Iterative Cost-Model Refresh** — Ji et al., 2024. (Cross-referenced from §1.) Retrains reward/cost models across three rounds of Safe RLHF using red-team data collected from the previous round's model. https://arxiv.org/abs/2406.15513
86. **MORLAIF (Multi-Objective RLAIF)** — 2024. (Cross-referenced from §2.) Trains one preference model per constitutional principle instead of a single blended one, to avoid principles washing each other out. https://arxiv.org/abs/2406.07295

## 12. Additional Scalable-Oversight & Amplification Variants

87. **Scalable Agent Alignment via Reward Modeling** — Leike et al., 2018 (DeepMind). (Full paper behind Recursive Reward Modeling, §3.) Proposes the general research direction of nesting reward-modeling agents to reach tasks too complex for direct human evaluation. https://arxiv.org/abs/1811.07871
88. **On 'Constitutional' AI (critical analysis)** — Digicon, 2023. Not a new method, but a widely-cited critique of CAI's "constitutional" framing worth reading alongside CAI itself when evaluating the technique's actual safety guarantees. https://digi-con.org/on-constitutional-ai/
89. **AI Alignment Strategies from a Risk Perspective** — 2025. Surveys RLHF and related learning-from-feedback safety mechanisms as one node in a broader map of alignment strategies and their shared failure modes. https://arxiv.org/abs/2510.11235
90. **A Survey of Process Reward Models** — 2025. Consolidated survey of PRM-based (process-supervision) methods, most of which trace back to Let's Verify Step by Step and feed into RLVR-based safety reasoning work in §5–6. https://arxiv.org/abs/2510.08049

---

