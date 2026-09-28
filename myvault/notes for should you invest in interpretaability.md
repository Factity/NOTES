# Monitoring Methods in Mathematical Notation

We describe the monitoring methods from the text using mathematical notation. We assume a language model $M$ with residual stream activations. For an input sequence $x$ (prompt + possibly generated tokens), let $R_l(x)[t] \in \mathbb{R}^d$ denote the residual stream activation at layer $l$ and token position $t$. The monitor $D$ takes activations (possibly aggregated across layers/tokens) and outputs a real-valued score $D(R(x))$.

## Two Monitoring Modes

- **Input‑based:** use only the final prompt token, i.e. $R_l(x)[\text{len}(x)-1]$ for each layer $l$.
- **Generation‑based:** use all tokens of the generation, i.e. $R_l(x)[\text{len}(x):]$ for each layer $l$.

Most monitors compute scores per layer and per token, then aggregate by taking the mean over layers and sequence positions.

---

## 1. Supervised Probes

Given a training set of positive examples (concept present) and negative examples (concept absent), we extract activations (e.g. at a chosen layer and token position). Let $\mathbf{z}_i \in \mathbb{R}^d$ be the feature vector for sample $i$.

### Mean Difference Probe

Compute the mean of positive and negative activation vectors:

$$
\mu_{\text{pos}} = \frac{1}{N_{\text{pos}}}\sum_{i:y_i=1} \mathbf{z}_i, \qquad
\mu_{\text{neg}} = \frac{1}{N_{\text{neg}}}\sum_{i:y_i=0} \mathbf{z}_i.
$$

The probe is the difference vector:

$$
\mathbf{w} = \mu_{\text{pos}} - \mu_{\text{neg}} \in \mathbb{R}^d.
$$

For a new sample with activation $\mathbf{z}$, the score is the dot product:

$$
D(\mathbf{z}) = \mathbf{w}\cdot \mathbf{z} = \sum_{j=1}^d w_j z_j.
$$

No bias term is used.

### Logistic Regression Probe

Learn a linear classifier with parameters $\mathbf{w}\in\mathbb{R}^d$ and $b\in\mathbb{R}$ by minimizing the logistic loss on training data. The predicted probability that a sample is positive is:

$$
p(y=1 \mid \mathbf{z}) = \sigma(\mathbf{w}\cdot \mathbf{z} + b) = \frac{1}{1+\exp(-(\mathbf{w}\cdot \mathbf{z} + b))}.
$$

The monitor score can be the logit or the probability:

$$
D(\mathbf{z}) = \mathbf{w}\cdot \mathbf{z} + b \quad \text{or} \quad D(\mathbf{z}) = p(y=1 \mid \mathbf{z}).
$$

Parameters are optimized via gradient descent on the negative log‑likelihood.

### MLP Probe

A multi‑layer perceptron with one or more hidden layers and non‑linear activations (e.g. ReLU). For a two‑layer MLP:

$$
\mathbf{h} = \text{ReLU}(W_1 \mathbf{z} + \mathbf{b}_1), \qquad
D(\mathbf{z}) = W_2 \mathbf{h} + b_2.
$$

The output is a real‑valued score; training uses cross‑entropy loss.

---

## 2. Out‑of‑Distribution (OOD) Detection

OOD detectors are trained only on “typical” (negative) examples. They assign a score that measures how unusual a new activation is; high score indicates OOD (potential concept present).

### Mahalanobis Distance Detector

Fit a multivariate Gaussian to the negative training activations:

$$
\mu = \frac{1}{N_{\text{neg}}}\sum_{i:y_i=0} \mathbf{z}_i, \qquad
\Sigma = \frac{1}{N_{\text{neg}}}\sum_{i:y_i=0} (\mathbf{z}_i - \mu)(\mathbf{z}_i - \mu)^\top.
$$

For a new activation $\mathbf{z}$, the squared Mahalanobis distance is:

$$
d_M^2(\mathbf{z}) = (\mathbf{z} - \mu)^\top \Sigma^{-1} (\mathbf{z} - \mu).
$$

The OOD score is often the negative log‑likelihood under the Gaussian (equivalent up to constants):

$$
D_{\text{OOD}}(\mathbf{z}) = -\log \mathcal{N}(\mathbf{z}; \mu, \Sigma) = \frac{d}{2}\log(2\pi) + \frac{1}{2}\log|\Sigma| + \frac{1}{2} d_M^2(\mathbf{z}).
$$

Higher score means more anomalous.

### Variational Autoencoder (VAE)

A VAE learns a probabilistic encoder $q_\phi(\mathbf{z}|\mathbf{x})$ and decoder $p_\theta(\mathbf{x}|\mathbf{z})$. For an input activation $\mathbf{x}$ (e.g. the residual stream vector), the ELBO (Evidence Lower Bound) is:

$$
\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\big),
$$

where $p(\mathbf{z})$ is a prior (usually $\mathcal{N}(0,I)$). The ELBO is higher for in‑distribution samples. As OOD score one often uses the negative ELBO:

$$
D_{\text{OOD}}(\mathbf{x}) = -\text{ELBO}(\mathbf{x}).
$$

---

## 3. Sparse Autoencoders (SAEs)

SAEs are unsupervised feature extractors. They encode an activation $\mathbf{x}\in\mathbb{R}^d$ into a sparse latent vector $\mathbf{z}\in\mathbb{R}^m$ (often $m \gg d$) using a linear map and a non‑linearity, and decode back to reconstruct the input.

A typical SAE (with ReLU sparsity) is defined by:

$$
\mathbf{z} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \qquad
\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}},
$$

with training objective combining reconstruction loss and a sparsity penalty on $\mathbf{z}$ (e.g. L1 penalty):

$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{z}\|_1.
$$

After training, the latent codes $\mathbf{z}$ can be used in two ways:

1. **Train linear probes on SAE latents:** treat $\mathbf{z}$ as features and train a logistic regression or mean‑difference probe as above.
2. **Use latents directly as classifiers:** a single latent dimension (or a small set) can act as a detector for a concept if it is found to be interpretable and aligned with that concept. For example, the score could be the value of a particular latent $z_j$, or a linear combination $D(\mathbf{x}) = \mathbf{w}\cdot \mathbf{z}$.

---

## 4. Brief Mention: Circuit Breakers and Latent Adversarial Training

These are **not** monitoring methods per se, but they share the goal of robustly detecting or preventing harmful outputs.  

- **Circuit Breakers** modify the model so that certain internal states trigger a refusal; they often train a classifier inside the model to detect harmful intent and then steer the generation away.
- **Latent Adversarial Training** encourages the model’s internal representations to be robust against adversarial perturbations that would otherwise fool a monitor.

Their mathematical formulations are more involved and are not detailed in the provided text; they are mentioned as related approaches.

---

All methods produce a scalar score $D(R(x))$. In practice, one may compute scores at multiple layers and token positions and average them, e.g.

$$
D_{\text{final}}(x) = \frac{1}{L \cdot T} \sum_{l=1}^L \sum_{t=1}^T D\big(R_l(x)[t]\big).
$$

This summarizes the numerical and mathematical aspects of the monitoring techniques described.


## 2. Ordinary Attacks Evaluation (§3.2)

We now formalize how the monitors are evaluated against standard jailbreaking methods.

### 2.1 Dataset Construction

Start with a dataset $\mathcal{D}_{\text{harmful}}$ of harmful prompts and a dataset $\mathcal{D}_{\text{benign}}$ of benign prompts. Use jailbreaking algorithms (GCG, PAIR, MSJ, AutoDAN, etc.) to produce successful jailbroken prompts from $\mathcal{D}_{\text{harmful}}$. Denote the resulting set of jailbroken prompts as $\mathcal{D}_{\text{jailbreak}}$. For each prompt in $\mathcal{D}_{\text{benign}}$ and $\mathcal{D}_{\text{jailbreak}}$, we obtain the model’s activations (possibly over multiple layers and tokens) and compute a monitor score $D(R(x))$.

Crucially, the monitors (probes, OOD detectors, SAE probes) are trained only on $\mathcal{D}_{\text{benign}}$ and on the original $\mathcal{D}_{\text{harmful}}$ (without jailbreaks). They are not trained on $\mathcal{D}_{\text{jailbreak}}$. Thus the evaluation measures transfer to unseen attack types.

### 2.2 Evaluation Metrics

Let

$$
S_{\text{benign}} = \{ D(x) : x \in \mathcal{D}_{\text{benign}}^{\text{test}} \}
$$

be the scores on a held‑out benign test set, and

$$
S_{\text{jail}} = \{ D(x) : x \in \mathcal{D}_{\text{jailbreak}}^{\text{test}} \}
$$

be the scores on the jailbroken harmful set. We treat benign as negative (label 0) and jailbroken harmful as positive (label 1).

**False Positive Rate (FPR):** fraction of benign samples incorrectly flagged as positive.

**True Positive Rate (TPR) / Recall:** fraction of jailbroken harmful samples correctly flagged.

Given a threshold $\tau$, the decision rule is:

$$
\hat{y} = 
\begin{cases}
1 & \text{if } D(x) \ge \tau, \\
0 & \text{otherwise.}
\end{cases}
$$

Then

$$
\text{FPR}(\tau) = \frac{|\{ s \in S_{\text{benign}} : s \ge \tau \}|}{|S_{\text{benign}}|},
\qquad
\text{TPR}(\tau) = \frac{|\{ s \in S_{\text{jail}} : s \ge \tau \}|}{|S_{\text{jail}}|}.
$$

To report “recall at 1% FPR”, choose the threshold $\tau^*$ such that $\text{FPR}(\tau^*) = 0.01$ (or the smallest threshold with FPR ≤ 0.01). Then the reported recall is $\text{TPR}(\tau^*)$.

The results in §3.2 state that logistic regression and MLP probes achieve ≥90% recall at 1% FPR, especially when using generation‑based activations, indicating that ordinary jailbreaks do not strongly obfuscate activations.

# Mathematical Explanation of Obfuscation Attack Formulation and Results

## 1. Obfuscation Attack Loss

Let $M$ be the language model, $D$ a monitor that takes residual stream activations $R(x)$ and outputs a real‑valued score, and $R(\cdot)$ the residual stream function. For a harmful request $\text{inp}$ and a target harmful response $\text{out}$, an attacker seeks a suffix $w$ (a sequence of tokens or embeddings) such that when appended to the input, the model still produces $\text{out}$ (jailbreak) but the monitor assigns a low (benign) score to the resulting activations.

Define the **obfuscation loss**:

$$
L(w) = \frac{1}{|\mathcal{D}|} \sum_{(\text{inp},\text{out}) \in \mathcal{D}} 
\Big[ \underbrace{\text{CrossEntropy}\big(M(\text{inp} \oplus w),\, \text{out}\big)}_{\text{Behavior Loss}} 
+ \underbrace{D\big(R(\text{inp} \oplus w \oplus \text{out})\big)}_{\text{Obfuscation Loss}} \Big] .
$$

- $\oplus$ denotes concatenation along the sequence dimension.
- $\mathcal{D}$ is a dataset of harmful request–response pairs. For a **universal suffix**, $\mathcal{D}$ is the entire harmful dataset; for an **input‑specific suffix**, $\mathcal{D}$ contains only that single request.

### 1.1 Terms Explained

- **Behavior Loss**  
  $\text{CrossEntropy}\big(M(\text{inp} \oplus w),\, \text{out}\big)$ measures how likely the model is to output the target response $\text{out}$ given the modified prompt $\text{inp} \oplus w$. Minimizing this term preserves the jailbreak: the model is forced to generate the harmful content.

- **Obfuscation Loss**  
  $D\big(R(\text{inp} \oplus w \oplus \text{out})\big)$ is the monitor’s score on the activations of the full sequence (prompt + suffix + response). Minimizing this term makes the monitor classify the sequence as benign, i.e., the activations appear “obfuscated” to the monitor.

The attacker minimizes $L(w)$ so that both objectives are achieved simultaneously: the model outputs the harmful response, yet the monitor is fooled.

---

## 2. Optimization for Different Attack Types

### 2.1 Embedding Suffixes (White‑box)

The suffix $w$ is a sequence of $n$ continuous embedding vectors in $\mathbb{R}^{d_{\text{emb}}}$, with fixed length $n=16$. Because $w$ is continuous, gradients of $L$ with respect to $w$ can be computed through the model and monitor. The attacker uses the **Adam optimizer** to solve

$$
w^* = \arg\min_w L(w).
$$

### 2.2 Hard Prompt Suffixes (Gray‑box)

The suffix $w$ consists of discrete tokens. The attacker has only **gray‑box access**: they can query the model’s logits and the monitor’s output, but cannot backpropagate gradients. Optimization is gradient‑free, using a modified version of the FLRT optimizer:

- The worst candidate string in the attack buffer is replaced instead of the best.
- The perplexity penalty is removed (since the goal is attacking latent monitors rather than producing natural text).

The FLRT optimizer can add or remove tokens, so the suffix length is not fixed. The objective remains the same $L(w)$, but evaluated via black‑box queries.

---

## 3. Evaluation Results and Transfer

### 3.1 Performance on Clean vs. Obfuscated Inputs

Let $\mathcal{S}_{\text{benign}}$ be scores on benign test samples and $\mathcal{S}_{\text{harm}}$ be scores on harmful test samples (with or without obfuscation). For a fixed false positive rate (FPR) $\alpha$ (e.g., 1%), the threshold $\tau$ is chosen such that

$$
\text{FPR}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{benign}} : s \ge \tau\}|}{|\mathcal{S}_{\text{benign}}|} \le \alpha .
$$

The recall (true positive rate) is then

$$
\text{Recall}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{harm}} : s \ge \tau\}|}{|\mathcal{S}_{\text{harm}}|} .
$$

The reported results state:

- On **clean prompts** (harmful without any suffix), all probes achieve high recall (e.g., >90% at 1% FPR).
- Against **obfuscation attacks** (suffixes optimized to minimize the monitor score), all probes drop to **0% recall** at the same FPR. This means the monitor scores for obfuscated harmful inputs are statistically indistinguishable from benign scores.

### 3.2 Transfer Between Monitors

Let $w$ be an adversarial suffix trained against monitor $D_A$ by minimizing $L_{D_A}(w)$. To test transfer, evaluate the same suffix $w$ against a different monitor $D_B$:

$$
\text{Bypass rate} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\text{inp},\text{out}) \in \mathcal{D}_{\text{test}}} \mathbb{1}\big[ D_B\big(R(\text{inp} \oplus w \oplus \text{out})\big) < \tau_B \big],
$$

where $\tau_B$ is the threshold for monitor $B$ at a fixed FPR. A high bypass rate indicates that the suffix also fools $B$ even though it was optimized for $A$.

Observed transfer patterns:

- **Probe → Probe:** Training against a logistic regression probe also bypasses MLP probes.
- **Generation‑based → Input‑based:** Suffixes trained against generation‑based probes transfer to input‑based probes.
- **Input‑based → Generation‑based:** No transfer in this direction.
- **Probe → OOD detector:** No transfer.
- **OOD detector → Input‑based probe:** Transfer is observed—training against an OOD detector bypasses input‑based probes, despite the very different underlying method.

These results indicate shared weaknesses among different latent‑space monitors, especially those that rely on similar input representations, and suggest that OOD detectors capture a more general distributional property that can also fool linear input‑based probes.

---

This formalization captures the mathematical essence of the obfuscation attack and the reported numerical findings.

# Evaluation Methodology Formalized (§2.2)

## 1. Dual Goal of the Attacker

The attacker aims to achieve two objectives simultaneously:

1. **Behavior goal:** induce the model to produce a target output (e.g., harmful response, correct SQL code).
2. **Obfuscation goal:** make the resulting activations appear benign to a latent‑space monitor $D$.

This creates a possible tradeoff: a more reliable jailbreak may produce more consistently harmful outputs, which could be easier for a harmfulness monitor to detect. Therefore, both aspects are evaluated separately.

---

## 2. Evaluating Behavior

### 2.1 Harmfulness (Jailbreak) Evaluation

Use **StrongREJECT** [Souly et al., 2024], which provides a score indicating how harmful a model output is. Formally, for a model output $y$ given an input $x$, StrongREJECT returns a scalar $S(y) \in [0,1]$, where higher values indicate more harmful content. Often a threshold is applied to decide if the attack succeeded (e.g., $S(y) > 0.5$). The **attack success rate (ASR)** for a set of prompts $\mathcal{D}$ is:

$$
\text{ASR} = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \mathbb{1}\big[ S\big(M(x)\big) \ge \tau_S \big],
$$

where $\tau_S$ is a chosen threshold.

### 2.2 SQL Code Correctness Evaluation

When the intended behavior is to generate correct SQL code (§5), the evaluation is different. Given a generated SQL query $q$ (the model output), it is executed against a database, and the returned entries are compared with the ground truth. Let $\text{result}(q)$ be the set of rows/entries returned by executing $q$, and $\text{truth}$ be the correct answer. The behavior success indicator is:

$$
\mathbb{1}\big[ \text{result}(q) = \text{truth} \big].
$$

The success rate is the average over test examples.

---

## 3. Evaluating Obfuscation

Let $D$ be a monitor that outputs a real‑valued score $D(R(x))$ for input $x$. We assume higher scores indicate the positive class (e.g., harmful). We have two sets:

- $\mathcal{X}_{\text{neg}}$: negative samples (e.g., benign).
- $\mathcal{X}_{\text{pos}}$: positive samples (e.g., harmful or jailbroken).

Compute the monitor scores for all samples:

$$
s_x = D(R(x)).
$$

### 3.1 Fixed False Positive Rate Threshold

We wish to choose a decision threshold $\tau$ such that the false positive rate (FPR) on negatives is at most $\alpha = 0.01$ (1%). The FPR at threshold $\tau$ is:

$$
\text{FPR}(\tau) = \frac{1}{|\mathcal{X}_{\text{neg}}|} \sum_{x \in \mathcal{X}_{\text{neg}}} \mathbb{1}[s_x \ge \tau].
$$

We select $\tau$ as the smallest threshold satisfying $\text{FPR}(\tau) \le 0.01$. In practice, this is equivalent to taking the $(1-\alpha)$‑quantile of the negative scores.

### 3.2 Recall (True Positive Rate) at That Threshold

The recall (or true positive rate, TPR) at the chosen threshold $\tau$ is:

$$
\text{Recall}(\tau) = \text{TPR}(\tau) = \frac{1}{|\mathcal{X}_{\text{pos}}|} \sum_{x \in \mathcal{X}_{\text{pos}}} \mathbb{1}[s_x \ge \tau].
$$

This measures the fraction of positive samples that the monitor correctly flags as positive while allowing at most 1% false alarms on negatives.

**Why this metric?**  
Unlike AUROC, which averages over all thresholds, recall at 1% FPR focuses on the low‑FPR regime, which is often the practically relevant operating point (e.g., a chatbot should not refuse benign requests too often).

---

## 4. Bootstrapped Confidence Intervals

Because the threshold $\tau$ and the recall depend on finite samples, we estimate uncertainty via bootstrapping.

For $b = 1, \dots, B$ (e.g., $B=1000$):

1. Resample with replacement from $\mathcal{X}_{\text{neg}}$ to form $\mathcal{X}_{\text{neg}}^{(b)}$ (same size as original).
2. Resample with replacement from $\mathcal{X}_{\text{pos}}$ to form $\mathcal{X}_{\text{pos}}^{(b)}$.
3. Using the resampled negative scores, compute the threshold $\tau^{(b)}$ that gives FPR ≤ 0.01.
4. Compute the recall on the resampled positive set:
   $$
   \text{Recall}^{(b)} = \frac{1}{|\mathcal{X}_{\text{pos}}^{(b)}|} \sum_{x \in \mathcal{X}_{\text{pos}}^{(b)}} \mathbb{1}\big[ D(R(x)) \ge \tau^{(b)} \big].
   $$

This yields a distribution of recall values. The 95% confidence interval is obtained by taking the 2.5th and 97.5th percentiles of $\{\text{Recall}^{(b)}\}_{b=1}^B$.

---

## 5. Sensitivity and Caveats

In some cases, recall is highly sensitive to the exact FPR chosen: e.g., recall at 2% FPR might be much higher than at 1% FPR. This sensitivity usually leads to large bootstrapped error bars. The text explicitly mentions such cases where applicable.

---

This mathematical framework captures the dual evaluation (behavior + obfuscation), the use of recall at 1% FPR, and the bootstrap procedure for confidence intervals as described in §2.2.