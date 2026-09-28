## 1. Setup

Let $\mathcal{D} = \{D_1, D_2, \dots, D_N\}$ be the collection of training datasets, where:

- $D_1$ is the **core** (general-purpose, retain) dataset.
- $D_2, \dots, D_N$ are the $N-1$ **auxiliary** (capability-specific, potentially forgettable) datasets.

A **capability profile** is a subset $S \subseteq \{2, \dots, N\}$ indicating which auxiliary capabilities are **retained** at inference time. The **forget set** is $F = \{2, \dots, N\} \setminus S$.

Let $M$ denote a model and $\ell(M, D_i)$ its validation cross-entropy loss on dataset $D_i$.

---

## 2. Parameter Partitioning

GRAM splits the model parameters into $N$ disjoint subsets:

$$
\Theta = \Theta_{\text{core}} \;\cup\; \bigcup_{i=2}^{N} \Theta_{\text{aux}}^{(i)}
$$

where the partition is **disjoint**: $\Theta_{\text{core}} \cap \Theta_{\text{aux}}^{(i)} = \emptyset$ and $\Theta_{\text{aux}}^{(i)} \cap \Theta_{\text{aux}}^{(j)} = \emptyset$ for $i \neq j$.

**What each subset contains:**

- $\Theta_{\text{core}}$: embeddings, all attention layers, all layer norms, unembedding, and the **core MLP** in every Transformer block.
- $\Theta_{\text{aux}}^{(i)}$: the $i$-th **auxiliary module** (a small MLP or LoRA adapter) in every block.

Each subset has its own AdamW optimizer state.

---

## 3. The MLP Block

In a standard dense Transformer, each layer has an MLP $f_{\text{mlp}}: \mathbb{R}^d \to \mathbb{R}^d$. GRAM replaces this with a **gateless mixture**:

For layer $\ell$, define experts $E_0^{(\ell)}, E_1^{(\ell)}, \dots, E_{N-1}^{(\ell)}$ where:

- $E_0^{(\ell)}$ is the **core MLP** with hidden dimension $d_{\text{core}}$.
- $E_{i}^{(\ell)}$ for $i \geq 1$ is the auxiliary module for dataset $D_{i+1}$ with hidden dimension $d_{\text{aux}} \ll d_{\text{core}}$.

The layer output for input $x \in \mathbb{R}^{B \times T \times d}$ given forward mask $\mathbf{m}^{\text{fwd}} \in \{0,1\}^N$ is:

$$
\text{MoE}(x; \mathbf{m}^{\text{fwd}}) = \sum_{j=0}^{N-1} m^{\text{fwd}}_j \cdot E_j^{(\ell)}(x)
$$

---

## 4. Forward and Backward Masks

For each training batch, GRAM specifies two binary masks over the $N$ parameter subsets:

- **Forward mask:** $\mathbf{m}^{\text{fwd}} \in \{0,1\}^N$ — which experts are active in the forward pass.
- **Backward mask:** $\mathbf{m}^{\text{bck}} \in \{0,1\}^N$ — which experts accumulate gradients.

The **freeze mechanism** is the mathematical heart of gradient routing. For expert $j$ with parameters $\theta_j$, the effective forward computation is:

$$
\tilde{E}_j(x) = 
\begin{cases}
E_j(x; \theta_j) & \text{if } m^{\text{bck}}_j = 1 \\[6pt]
E_j(x; \tilde{\theta}_j) & \text{if } m^{\text{bck}}_j = 0
\end{cases}
$$

where $\tilde{\theta}_j = \text{stop\_grad}(\theta_j)$. Activations flow through, but $\frac{\partial \tilde{E}_j}{\partial \theta_j} = 0$.

Thus the full forward pass is:

$$
h_{\text{out}} = \sum_{j=0}^{N-1} m^{\text{fwd}}_j \cdot \tilde{E}_j(\text{RMSNorm}(h_{\text{in}}))
$$

and only experts with $m^{\text{bck}}_j = 1$ receive parameter updates.

---

## 5. Training: Batch-Dependent Routing

Three hyperparameters control batch composition:

- $p_{\text{af}}(i)$: **Auxiliary Factor** — sampling frequency of dataset $D_i$.
- $p_{\text{as}} \in [0,1]$: **Auxiliary Spread** — probability that core parameters update on an auxiliary batch.
- $p_{\text{cr}} \in [0,1]$: **Core Robustness** — probability that a random auxiliary module activates on a core batch.

### 5.1 Core Batch ($D_1$)

Sample a batch $b \sim D_1$. With probability $p_{\text{cr}}$, sample a random auxiliary index $k \sim \text{Uniform}\{2, \dots, N\}$.

**Forward mask:**
$$
m^{\text{fwd}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core always active})\\
1 & j = k \quad \text{with prob. } p_{\text{cr}} \\
0 & \text{otherwise}
\end{cases}
$$

**Backward mask:**
$$
m^{\text{bck}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core always updated})\\
1 & j = k \quad \text{if } m^{\text{fwd}}_k = 1 \\
0 & \text{otherwise}
\end{cases}
$$

All active parameters are updated. The random auxiliary activation forces the core to remain robust to the presence or absence of any single auxiliary.

### 5.2 Auxiliary Batch ($D_i$, $i \geq 2$)

Sample a batch $b \sim D_i$.

**Forward mask:**
$$
m^{\text{fwd}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core})\\
1 & j = i \quad (\text{auxiliary } i)\\
0 & \text{otherwise}
\end{cases}
$$

**Backward mask:**
$$
m^{\text{bck}}_j = 
\begin{cases}
1 & j = i \quad (\text{auxiliary } i \text{ always updated})\\
\mathbb{1}[\text{Bernoulli}(p_{\text{as}})] & j = 0 \quad (\text{core updated with prob. } p_{\text{as}})\\
0 & \text{otherwise}
\end{cases}
$$

When $p_{\text{as}} = 0$, the core is frozen and only $\Theta_{\text{aux}}^{(i)}$ learns $D_i$, maximizing modularity. When $p_{\text{as}} = 1$, the core absorbs auxiliary gradients, destroying modularity.

---

## 6. Inference and Capability Control

At inference, a capability profile $S \subseteq \{2, \dots, N\}$ is served by activating only the retained modules:

**Inference forward mask:**
$$
m^{\text{inf}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core always on})\\
1 & j \in S \\
0 & j \in F = \{2,\dots,N\} \setminus S \quad (\text{ablated})
\end{cases}
$$

The ablated model is:

$$
M_S = \text{Model}\Big(\Theta_{\text{core}} \cup \bigcup_{i \in S} \Theta_{\text{aux}}^{(i)}\Big), \qquad \Theta_{\text{aux}}^{(j)} \text{ for } j \in F \text{ is set to zero}
$$

---

## 7. Compute Ratio Metric

Let $M_{\text{BL}}$ be the baseline dense Transformer trained on $D_{\text{all}} = \bigcup_{i=1}^{N} D_i$.

For each dataset $D_i$, let $L_i: \mathbb{R}_{\geq 0} \to \mathbb{R}_{&gt;0}$ be the fitted power-law learning curve of the baseline, mapping training step $s$ to validation loss:

$$
L_i(s) = \ell\big(M_{\text{BL}}^{(s)}, D_i\big)
$$

where $M_{\text{BL}}^{(s)}$ is the baseline checkpoint at step $s$. Since $L_i$ is monotonic and continuous, its inverse $L_i^{-1}$ exists.

The **Compute Ratio** of model $M$ on dataset $D_i$ is:

$$
\boxed{
\text{CR}(M, D_i) = \frac{L_i^{-1}\big(\ell(M, D_i)\big)}{L_i^{-1}\big(\ell(M_{\text{BL}}, D_i)\big)}
}
$$

**Interpretation:**

| $\text{CR}(M, D_i)$ | Meaning |
|---|---|
| $= 1$ | $M$ matches the baseline on $D_i$ |
| $&gt; 1$ | $M$ is **better** than baseline (reaches same loss with less compute) |
| $&lt; 1$ | $M$ is **worse** than baseline (needs more compute to reach same loss) |

For a **retain** dataset ($i \in \{1\} \cup S$), higher CR is better. For a **forget** dataset ($i \in F$), lower CR is better — it means the ablated model has forgotten the capability.

---

## 8. The Optimization Objective (Implicit)

GRAM does not optimize a single scalar loss. Instead, it optimizes $N$ coupled objectives via the routing masks:

$$
\min_{\Theta} \; \mathbb{E}_{b \sim \mathcal{B}}\Big[ \mathcal{L}\big(M(b; \mathbf{m}^{\text{fwd}}), y_b\big) \Big]
$$

subject to the constraint that for each batch, gradients flow only to the experts selected by $\mathbf{m}^{\text{bck}}$. The expectation is over the mixed batch distribution where $D_i$ is sampled with frequency governed by $p_{\text{af}}(i)$.

The separate AdamW optimizers effectively maintain independent momentum states $\{v_j\}_{j=0}^{N-1}$, and at step $t$:

$$
\theta_j^{(t+1)} = \theta_j^{(t)} - \eta \cdot \frac{\hat{v}_j^{(t)}}{\sqrt{\hat{s}_j^{(t)}} + \epsilon} \quad \text{iff } m^{\text{bck}}_j = 1 \text{ at step } t
$$

where $\hat{v}_j, \hat{s}_j$ are the bias-corrected first and second moment estimates for optimizer $j$.