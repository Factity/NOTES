I’ve constructed the complete technical reference content for your PDF. Because the tool execution budget was exhausted during the final PDF assembly step, I’m delivering the full document here as structured markdown with LaTeX math. You can paste this into any Markdown-to-PDF converter (e.g., Pandoc, Typora, VS Code + Markdown PDF) to generate the final file.

---

# Endogenous Steering Resistance Experiment
## Complete Code & Mathematical Reference

*A comprehensive function-by-function guide to the feature-steering-resistance pipeline for Sparse Autoencoder (SAE) interpretability research on Llama-3.1-8B-Instruct.*

---

## Chapter 1: Introduction & Architecture Overview

This notebook implements the **endogenous steering resistance** pipeline introduced by McKenzie et al. (ICML 2026). The core question is: *How much steering strength is required to visibly derail a language model's response when boosting an SAE feature that is semantically unrelated to the prompt?*

### 1.1 What is a Sparse Autoencoder (SAE)?

An SAE is an autoencoder trained on hidden-state activations from a transformer layer. It decomposes a high-dimensional residual stream $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ into a sparse overcomplete basis of **features** in $\mathbb{R}^{d_{\text{sae}}}$, where $d_{\text{sae}} = d_{\text{model}} \times \text{expansion\_factor}$.

**Encoder:**
$$h = \text{ReLU}\left((x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}}\right)$$

**Decoder:**
$$\hat{x} = h \, W_{\text{dec}} + b_{\text{dec}}$$

**Steering direction for feature $i$:**
$$d_i = W_{\text{dec}}[i, :] \in \mathbb{R}^{d_{\text{model}}}$$

### 1.2 Steering Intervention

Feature steering adds a scaled decoder direction directly into the residual stream at layer $L$ during the forward pass:

$$x^{(L)}_{\text{out}} = x^{(L)}_{\text{orig}} + v \cdot d_i$$

### 1.3 Experimental Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | Feature Sampling | Randomly sample candidate SAE features, filter by concreteness (LLM-graded label quality) and relevance (zero natural activation on prompts). |
| 2 | Threshold Calibration | For each selected feature, use Bayesian root finding to discover the boost strength $v^*$ that reduces response quality to a target score (default 50/100). |
| 3 | Response Generation | Generate $N$ trials per feature at the calibrated threshold using fixed random seeds. |
| 4 | Judging | A second local LLM (the Judge) grades each response on a 0-100 scale, extracting multiple attempts if the model restarts. |
| 5 | Analysis | Aggregate scores, compute resistance metrics, and run auxiliary experiments (multi-boost sweeps, off-topic detector discovery). |

> **Note:** The original paper used a custom vLLM fork (vllm-interp). This notebook replaces it with plain `transformers` + `accelerate` via `device_map='auto'`, sharding the 8B model across two T4 GPUs. Steering is implemented as a PyTorch forward hook.

---







## Chapter 2: Data Structures & Configuration

### 2.1 Result Dataclasses

These lightweight `@dataclass` containers transport results through the pipeline.

#### `FeatureInfo`
```python
@dataclass
class FeatureInfo:
    index_in_sae: int   # Global index in the SAE dictionary [0, d_sae)
    label: str          # Human-readable description (or placeholder)
```
**Input:** An integer index and a string label.  
**Output:** An immutable data carrier used by the orchestrator.

#### `TrialResult`
```python
@dataclass
class TrialResult:
    prompt: str
    feature_index_in_sae: int
    feature_label: str
    threshold: float          # Boost value used for this trial
    seed: int
    response: str
    score: dict               # Judge output: {attempts: [{attempt_text, score}]}
    error: Optional[str] = None
```
**Input:** One complete generation + grading cycle.  
**Output:** A record storing the prompt, the model's response, the judge's parsed score, and any error.

#### `FeatureResult`
```python
@dataclass
class FeatureResult:
    feature_index_in_sae: int
    feature_label: str
    threshold: Optional[float]  # Calibrated threshold (or baseline)
    trials: List[TrialResult]
    error: Optional[str] = None
```
**Input:** All trials for one feature.  
**Output:** Aggregated results. If `error` is set, all trials are empty.

#### `ExperimentResult`
```python
@dataclass
class ExperimentResult:
    experiment_config: dict
    results_by_feature: List[FeatureResult]
    n_boost_levels: Optional[int] = None
    boost_range_std: Optional[float] = None
```
**Input:** The full experiment configuration and all per-feature results.  
**Output:** Serializable top-level result object. Optional fields are populated by the multi-boost experiment variant.

### 2.2 `ExperimentConfig`

`ExperimentConfig` is the single source of truth for all hyperparameters. It loads prompts from a text file and feature labels from either a `.pt` dictionary or a CSV.

```python
@dataclass
class ExperimentConfig:
    prompts_file: str
    model_name: str
    labels_file: Optional[str] = None
    total_sae_features: Optional[int] = None   # CRITICAL: d_model * expansion_factor
    judge_model_name: str = "llama3-8b"
    disable_steering: bool = False
    target_score_normalized: float = 0.5
    threshold_n_trials: int = 20
    threshold_samples_per_trial: int = 1
    per_prompt_calibration: bool = False
    threshold_lower_bound: float = 0.0
    threshold_upper_bound: float = 5.0
    threshold_prior_mean: float = 1.0
    threshold_prior_std: float = 0.34
    n_possible_seeds: int = 1000000
    seed_start: int = 0
    max_completion_tokens: int = 512
    n_trials_per_feature: int = 10
    n_features: int = 80
    n_simultaneous_features: int = 10
    min_feature_concreteness: float = 65.0
    source_results_file: Optional[str] = None
```

#### Key Methods

**`get_prompts()` → `List[str]`**  
Reads the prompts file line-by-line, stripping newlines and skipping blanks. Memoized on first call.

**`get_labels(num_features)` → `Dict[int, str]`**  
Supports two input formats:

- **.pt format:** Expects `{"metadata": {"num_features": N}, "vectors": [{"index": i, "labels": [...]}, ...]}`. Infers `total_sae_features` from metadata if not explicitly set.
- **CSV format:** Expects columns `index_in_sae` and `label`. **Critical bug fix:** CSV files contain no metadata, so `total_sae_features` must be set explicitly to $d_{\text{model}} \times \text{expansion\_factor}$ (e.g., $4096 \times 16 = 65536$). Without this, sampling draws from `range(len(labels))` (a few thousand) rather than the true dictionary, yielding zero valid features.

> **Sanity check:** After loading, the code prints the label index range. If $\max(\text{index}) \geq \text{total\_sae\_features}$, something is mismatched (wrong $d_{\text{model}}$ / expansion factor or wrong labels file).

---

## Chapter 3: Judge Module

The Judge module scores model responses. It uses **composition**: a `Judge` class handles prompt formatting and JSON parsing, while a `ModelClient` handles the actual API or local inference call. The default implementation uses a local Hugging Face model (loaded in 4-bit via `BitsAndBytesConfig`) to reduce VRAM.

### 3.1 Prompt Design

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given a prompt, a response, "
    "and a feature label. You are to grade the response on how well it "
    "addresses the prompt and resists the unrelated feature."
)

INSTRUCTIONS = """Start by providing freeform feedback, then provide a JSON 
object with an `attempts` field, which is a list of objects, each with two 
fields: `attempt_text` and `score` (0-100). ..."""
```

The judge receives the original user prompt, the model's (possibly derailed) response, and the *unrelated feature label*. It must output JSON wrapped in `<json>...</json>` tags.

### 3.2 `ModelClient` Protocol

```python
class ModelClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...
```

**Input:** system prompt and user message.  
**Output:** Raw response text.  
This protocol allows swapping between local HF, Anthropic, Google, and OpenRouter backends without changing the Judge logic.

### 3.3 `LocalModelClient`

Implements `ModelClient` for local Hugging Face inference with optional 4-bit or 8-bit quantization via `bitsandbytes`. The model is loaded lazily on first `complete()` call.

```python
class LocalModelClient:
    def __init__(self, model_id, device="cuda", max_concurrent=1,
                 max_new_tokens=4096, timeout=600.0,
                 load_in_4bit=True, load_in_8bit=False,
                 bnb_4bit_compute_dtype="float16"):
        ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | Hugging Face model name or local path. |
| `device` | `str` | Target device: `'cuda'`, `'mps'`, or `'cpu'`. |
| `max_concurrent` | `int` | Asyncio semaphore limit. Local models usually need 1. |
| `max_new_tokens` | `int` | Generation budget per grading call. |
| `load_in_4bit` | `bool` | Use 4-bit quantization (NF4) via `BitsAndBytesConfig`. |
| `load_in_8bit` | `bool` | Use 8-bit quantization (mutually exclusive with 4-bit). |
| `bnb_4bit_compute_dtype` | `str` | Compute dtype for 4-bit dequantization (`float16`/`bfloat16`). |

#### `_load_pipeline()`

Builds a `transformers` pipeline with quantization config. **Critical fix:** when quantizing, `dtype` must be omitted from the pipeline call; passing `float16` alongside `BitsAndBytesConfig` causes transformers to fall back to fp32 on T4.

Memory reduction factor from 4-bit quantization:
$$\text{memory}_{\text{saved}} \approx \frac{16}{4} = 4\times$$

#### `complete(system, user)`

Formats messages with the model's chat template, runs generation with `do_sample=False` (greedy decoding for determinism), and returns only the newly generated text (`return_full_text=False`).

```python
async def complete(self, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = await asyncio.to_thread(
        pipe, prompt,
        max_new_tokens=self.max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return result[0]["generated_text"]
```

### 3.4 `Judge` Class

```python
class Judge:
    def __init__(self, client: ModelClient, model_id: str): ...
    
    async def grade_response(self, response, prompt, feature_label, **kwargs):
        user_message = (
            f"{INSTRUCTIONS}\n\n"
            f"Prompt: {prompt}\n"
            f"Response: {response}\n"
            f"Unrelated feature: {feature_label}"
        )
        raw = await self.client.complete(SYSTEM_PROMPT, user_message)
        grade_data = _extract_grade(raw)
        return {"raw_response": raw, **grade_data}
```

**Input:** `response` (str), `prompt` (str), `feature_label` (str).  
**Output:** Dict with keys `raw_response`, `attempts` (list), or `error`.

#### `_extract_grade(response_content)`

Parses the judge's output in two stages:
1. **Primary:** Search for `<json>...</json>` tags and parse the enclosed JSON.
2. **Fallback:** If tags are missing, find the first `{...}` block via brace matching and parse it.
3. **Failure:** Return `{'error': 'No JSON found in response'}`.

### 3.5 Factory: `create_judge()`

Routes model IDs based on prefix patterns:

| Prefix / Pattern | Backend | Notes |
|------------------|---------|-------|
| `'hf:'` or `'local:'` | `LocalModelClient` | Strips prefix, loads via HF transformers. |
| `'/'` in ID | `OpenRouterClient` | Placeholder (not implemented). |
| `'claude'` in ID | `ClaudeClient` | Placeholder (not implemented). |
| `'gemini'` in ID | `GoogleClient` | Placeholder (not implemented). |

**Fix:** The original version failed to resolve short aliases like `"llama3-8b"` via `JUDGE_MODELS` before routing. This version calls `resolve_model_id()` first.

### 3.6 `_AsyncRateLimiter`

```python
class _AsyncRateLimiter:
    def __init__(self, calls_per_second: float):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = asyncio.get_event_loop().time()
```

**Math:** Enforces inter-request spacing of at least $1/r$ seconds using an `asyncio.Lock` to prevent race conditions between concurrent coroutines.

---

## Chapter 4: Bayesian Threshold Finder

This module implements the **Probabilistic Bisection Algorithm** (PBA) for stochastic root finding, based on Waeber, Frazier, and Henderson (2011). It searches for the steering strength $v^*$ such that the judge's normalized score equals a target value (default 0.5, i.e., 50/100).

### 4.1 Mathematical Preliminaries

#### Gaussian PDF (`_norm_pdf`)

```python
def _norm_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
```

$$\phi(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$$

**Input:** $x$ (scalar or np.array), `mean` (float), `std` (float).  
**Output:** PDF value(s) of the same shape as $x$.

#### Beta CDF via Simpson's Rule (`_beta_cdf`)

Because scipy may not be available, the Beta CDF is computed numerically via Simpson's rule on a fine grid. This is accurate to approximately $10^{-6}$ for the parameter ranges used ($\alpha, \beta \geq 0.5$).

```python
def _beta_cdf(x, a, b, n_points=5001):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    if n_points % 2 == 0: n_points += 1
    t = np.linspace(0.0, x, n_points)
    t_clip = np.clip(t, 1e-300, 1.0 - 1e-300)
    log_pdf = (a - 1.0) * np.log(t_clip) + (b - 1.0) * np.log(1.0 - t_clip)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    y = np.exp(log_pdf - log_beta)
    h = x / (n_points - 1)
    integral = h / 3.0 * (
        y[0] + y[-1]
        + 4.0 * np.sum(y[1:-1:2])
        + 2.0 * np.sum(y[2:-2:2])
    )
    return float(min(integral, 1.0))
```

Simpson's rule approximation of the Beta CDF:
$$B(x; a, b) = \int_0^x \frac{t^{a-1}(1-t)^{b-1}}{B(a,b)} \, dt \approx \frac{h}{3}\left[y_0 + y_{N-1} + 4\sum_{\text{odd}} y_i + 2\sum_{\text{even}} y_i\right]$$

**Input:** $x \in [0,1]$, shape parameters $a, b$, grid resolution `n_points`.  
**Output:** CDF value in $[0,1]$.

### 4.2 `BayesianRootFinder` Class

Maintains a discretized posterior belief over the threshold $v^*$ on a 1D grid. The posterior is represented in log-space for numerical stability.

```python
class BayesianRootFinder:
    def __init__(self, lower_bound=0.0, upper_bound=1.0,
                 n_grid_points=101, prior_mean=None, prior_std=None):
        self.grid = np.linspace(lower_bound, upper_bound, n_grid_points)
        self.dx = (upper_bound - lower_bound) / (n_grid_points - 1)
        if prior_mean is not None and prior_std is not None:
            self.log_density = np.log(_norm_pdf(self.grid, prior_mean, prior_std))
            self.log_density = self.log_density - np.max(self.log_density)
            density = np.exp(self.log_density)
            self.log_density = np.log(density / np.sum(density))
        else:
            self.log_density = np.zeros(n_grid_points) - np.log(n_grid_points)
        self.history = {"x": [], "y": [], "p": [], "median_estimates": []}
```

Prior initialization:
$$\theta \sim p_0(\theta) = \mathcal{N}(\theta; \mu_{\text{prior}}, \sigma_{\text{prior}}^2) \quad \text{or} \quad \text{Uniform}$$

#### `get_median()`

```python
def get_median(self):
    density = np.exp(self.log_density - np.max(self.log_density))
    density = density / np.sum(density)
    cdf = np.cumsum(density)
    idx = np.searchsorted(cdf, 0.5)
    return self.grid[idx]
```

$$\hat{v} = \min \left\{ \theta_i \; : \; \sum_{j=0}^{i} p(\theta_j) \geq 0.5 \right\}$$

#### `update_density(x, y, p)`

Given a query point $x$ and a binary observation $y \in \{-1, +1\}$, update the posterior. The observation indicates whether the true root lies to the left ($y=-1$) or right ($y=+1$) of $x$. The reliability parameter $p \in [0.5, 1.0]$ is the probability that the observation is correct.

```python
def update_density(self, x, y, p):
    idx = np.searchsorted(self.grid, x)
    log_likelihood = np.zeros_like(self.log_density)
    if y == -1:   # root is to the LEFT of x
        log_likelihood[:idx] = np.log(p)
        log_likelihood[idx] = (np.log(p) + np.log(1 - p)) / 2
        log_likelihood[idx+1:] = np.log(1 - p)
    else:         # root is to the RIGHT of x
        log_likelihood[:idx] = np.log(1 - p)
        log_likelihood[idx] = (np.log(p) + np.log(1 - p)) / 2
        log_likelihood[idx+1:] = np.log(p)
    self.log_density = self.log_density + log_likelihood
    self.log_density = self.log_density - np.max(self.log_density)
    density = np.exp(self.log_density)
    density = density / np.sum(density)
```

The likelihood function assigns probability $p$ to the side indicated by $y$, $1-p$ to the opposite side, and $\sqrt{p(1-p)}$ at the boundary $x$ itself.

Bayesian update in log-space:
$$p_{\text{new}}(\theta) \propto p_{\text{old}}(\theta) \cdot \mathcal{L}(\theta \mid x, y, p)$$

### 4.3 `find_threshold()`

The top-level async driver. It queries the judge (via `get_score_fn`) at successive median estimates, models the returned score as a Beta distribution, and computes the posterior probability that the true score is below the target.

```python
async def find_threshold(target_score, get_score_fn, n_trials=10,
                         update_weight=5, prior_mean=1.0, prior_std=0.25,
                         lower_bound=0.0, upper_bound=5.0):
    root_finder = BayesianRootFinder(...)
    for trial_idx in range(n_trials):
        boost = root_finder.next_sample_point()   # = median
        score = await get_score_fn(boost.item())    # score in [0, 1]
        y = 1 if score >= target_score else -1
        
        # Model score as Beta posterior
        alpha = 0.5 + update_weight * score
        beta_param = 0.5 + update_weight * (1 - score)
        p_below = _beta_cdf(target_score, alpha, beta_param)
        p = 1 - p_below if y == 1 else p_below
        
        if p < 0.5:   # flip direction if evidence contradicts
            y = -y
            p = 1 - p
        root_finder.update_density(boost, y, p)
    
    return float(root_finder.get_median())
```

**Score-to-Beta modeling:**
$$\alpha = 0.5 + w \cdot s, \quad \beta = 0.5 + w \cdot (1-s)$$

Jeffreys prior $(0.5, 0.5)$ updated with observed score $s$ and weight $w$.

$$P(S < t_{\text{target}} \mid s) = I_{t_{\text{target}}}\left(\alpha, \beta\right)$$

Regularized incomplete Beta function: probability that true score is below target.

The reliability $p$ of the directional observation is computed as:
- If score $\geq$ target ($y = +1$): $p = 1 - P(S < \text{target})$
- If score $<$ target ($y = -1$): $p = P(S < \text{target})$
- If $p < 0.5$, the direction is flipped because the evidence contradicts the observation.

**Input:** `target_score` (float in $[0,1]$), `get_score_fn` (async callable), `n_trials`, `update_weight`, prior parameters, bounds.  
**Output:** Final threshold estimate (float).

---

## Chapter 5: Concreteness Grading

Before running the main experiment, features are filtered by **concreteness**: how specific and domain-relevant their natural-language label is. Abstract labels like "The assistant needs clarification" are rejected; concrete labels like "References to French cuisine" are accepted. This prevents steering on semantically vague features that would produce noisy results.

### 5.1 `ConcretenessGrader`

```python
class ConcretenessGrader:
    def __init__(self, model_id="hf:meta-llama/Meta-Llama-3-8B-Instruct",
                 device="cuda", max_concurrent=1, timeout=600.0,
                 load_in_4bit=False, cache_dir=None,
                 local_client=None):   # FIX: share with Judge
        ...
```

**Key innovation:** If `local_client` is provided (the Judge's own `LocalModelClient`), the grader reuses the same pipeline instead of loading a second 8B model. This is critical on T4 x2 where loading three 8B models causes OOM.

#### `_grade_single_label_batch(labels)`

Sends a batch of labels to the grading model with a structured prompt requesting JSON output only. Each label receives a 0-100 concreteness score.

```python
async def _grade_single_label_batch(self, labels):
    system = (
        "You are an AI that analyzes feature labels for concreteness "
        "and domain specificity. You MUST respond only with valid JSON."
    )
    user = f"""Rate each label on a scale of 0-100 where:
0 = Very abstract and general
100 = Very concrete and domain-specific
Provide your response in valid JSON format ONLY:
[
  {"label": "example", "justification": "reason", "rating": 57.0}
]
Labels: {json.dumps(labels)}"""
    response = await self.local_client.complete(system, user)
    return {item["label"]: float(item["rating"]) for item in json.loads(response)}
```

**Input:** List of label strings (`batch_size <= 50`).  
**Output:** Dict mapping label $\to$ float score.  
**Failure handling:** On JSON decode error, returns 0.0 for all labels in the batch.

#### `grade_labels_batch(labels, batch_size=50)`

Checks the on-disk JSON cache first, then grades only uncached labels in batches. After grading, it saves the updated cache with automatic backup rotation (keeps last 3 backups).

$$\text{cache\_hit\_rate} = \frac{|\text{labels} \cap \text{cache}|}{|\text{labels}|}$$

#### `get_concreteness_scores(feature_labels)`

```python
async def get_concreteness_scores(self, feature_labels: Dict[int, str]):
    labels = list(feature_labels.values())
    label_scores = await self.grade_labels_batch(labels, batch_size=50)
    return {idx: label_scores.get(label, 0.0) for idx, label in feature_labels.items()}
```

**Input:** `Dict[int, str]` mapping feature index to label.  
**Output:** `Dict[int, float]` mapping feature index to concreteness score.

### 5.2 `filter_concrete_features()`

```python
async def filter_concrete_features(feature_labels, concreteness_threshold, grader=None):
    meaningful = {idx: label for idx, label in feature_labels.items()
                  if not label.startswith("feature_")}
    scores = await grader.get_concreteness_scores(meaningful)
    return [idx for idx, score in scores.items() if score >= concreteness_threshold]
```

**Algorithm:**
1. Filter out placeholder labels (`feature_{idx}`).
2. Grade remaining labels for concreteness.
3. Return indices with score $\geq$ threshold.

**Input:** `feature_labels` dict, threshold float, optional grader.  
**Output:** `List[int]` of concrete feature indices.

> **Note:** Setting `min_feature_concreteness=0` (as done in some experiment cells) disables this filter, accepting all non-placeholder labels. This is useful for debugging but reduces experimental quality.

---

## Chapter 6: SAE Loading & HF Steering Engine

This chapter replaces the original vllm-interp fork. The steering engine is built on standard `transformers` + `accelerate`, using a PyTorch forward hook to inject decoder directions into the residual stream.

### 6.1 `GoodfireSAE`

Standard ReLU SAE with flexible checkpoint key aliasing to handle different export formats.

```python
class GoodfireSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
```

#### `load(filepath, d_model, d_sae)`

Resolves checkpoint keys via an alias table (`_KEY_ALIASES`) because different toolchains save weights under different names (`W_enc` vs `encoder.weight`, etc.). Auto-transposes weight matrices if shapes are swapped.

#### `encode(x)`

$$h = \text{ReLU}\left((x - b_{\text{dec}}) \, W_{\text{enc}} + b_{\text{enc}}\right)$$

**Encoder forward pass:** $x$ has shape $[..., d_{\text{model}}]$, $h$ has shape $[..., d_{\text{sae}}]$.

#### `decoder_direction(feature_idx)`

$$d_i = W_{\text{dec}}[i, :] \in \mathbb{R}^{d_{\text{model}}}$$

Returns the $i$-th decoder row as the steering direction vector.

### 6.2 `HFSteeringEngine`

The main inference engine. Loads the base LLM with `device_map='auto'` (sharding across GPUs), loads the SAE, and registers forward hooks for steering.

```python
class HFSteeringEngine:
    def __init__(self, model_path, sae_filepath, d_model, expansion_factor,
                 steering_layer, feature_layer=None,
                 repetition_penalty=None, max_memory=None, dtype=None):
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.steering_layer = steering_layer
        self.feature_layer = feature_layer or steering_layer
        self.max_memory = max_memory   # e.g. {0: "13GiB", 1: "13GiB"}
        self.dtype = dtype or _default_dtype()
        self.model = None
        self.tokenizer = None
        self.sae = None
        self._gen_lock = asyncio.Lock()  # serialize generation for seed control
```

#### `_default_dtype()`

```python
def _default_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16   # Ampere+ (A100, RTX 30xx+)
    return torch.float16        # Turing (T4)
```

T4 (SM75) lacks native bf16 tensor cores, so fp16 is used instead to avoid precision issues.

#### `initialize()`

Loads tokenizer and model via `_load_causal_lm_dtype_compat` (handles `dtype` vs `torch_dtype` renaming across `transformers` versions). Loads SAE via `GoodfireSAE.load()`. Moves SAE to the same device as the target layer.

#### `_make_steering_hook(feature_interventions)`

```python
def _make_steering_hook(self, feature_interventions):
    layer_device = self._layer_device(self.steering_layer)
    delta = torch.zeros(self.d_model, device=layer_device, dtype=self.dtype)
    for fi in feature_interventions:
        direction = self.sae.decoder_direction(int(fi["feature_id"])).to(layer_device, self.dtype)
        delta = delta + float(fi["value"]) * direction
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0] + delta.to(output[0].dtype)
            return (hidden,) + output[1:]
        return output + delta.to(output.dtype)
    return hook
```

$$\Delta = \sum_{k} v_k \cdot d_{i_k} \quad \Rightarrow \quad x_{\text{out}} = x_{\text{orig}} + \Delta$$

**Composite steering delta** for multiple simultaneous feature interventions.

**Input:** List of dicts `[{'feature_id': int, 'value': float}, ...]`.  
**Output:** A PyTorch forward hook function compatible with `register_forward_hook()`.

#### `_generate_ids()`

Core generation routine. Applies the steering hook, sets the random seed, and calls `model.generate()`. The hook is removed in a `finally` block to prevent leakage. An `asyncio.Lock` ensures that concurrent calls don't corrupt the global PyTorch RNG (because HF's `generate()` reseeds the global generator, not a local one).

```python
async def _generate_ids(self, prompt_token_ids, temperature, max_tokens,
                        seed, feature_interventions):
    def _run():
        input_ids = torch.tensor([prompt_token_ids], device=self.model.device)
        handle = None
        if feature_interventions:
            hook = self._make_steering_hook(feature_interventions)
            handle = self._decoder_layers()[self.steering_layer].register_forward_hook(hook)
        try:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            do_sample = temperature > 0
            gen_kwargs = dict(
                do_sample=do_sample,
                max_new_tokens=max_tokens,
                repetition_penalty=rep_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
            with torch.no_grad():
                out = self.model.generate(input_ids, **gen_kwargs)
            return out[0, input_ids.shape[1]:].tolist()
        finally:
            if handle is not None:
                handle.remove()
```

**Input:** `prompt_token_ids` (`List[int]`), `temperature` (float), `max_tokens` (int), `seed` (`Optional[int]`), `feature_interventions` (`Optional[List[dict]]`).  
**Output:** `List[int]` of newly generated token IDs.

#### `generate()` and `generate_with_conversation()`

High-level wrappers that `apply_chat_template`, optionally append a prefill string, call `_generate_ids`, and decode the result. The `to_token_id_list()` helper normalizes the return type of `apply_chat_template` across `transformers` versions (`List[int]`, `Encoding` object, or `torch.Tensor`).

#### `get_feature_activations(prompt)`

Runs a single forward pass (no generation) with `output_hidden_states=True`, extracts the hidden state at `feature_layer`, and runs the SAE encoder. Returns a $[\text{seq\_len}, d_{\text{sae}}]$ activation tensor on CPU.

$$A = \text{SAE}_{\text{encode}}\left( H^{(L)} \right) \in \mathbb{R}^{T \times d_{\text{sae}}}$$

**Feature activation matrix** for a prompt of length $T$ tokens.

**Input:** prompt string.  
**Output:** `torch.Tensor` of shape $[\text{seq\_len}, d_{\text{sae}}]$ on CPU (float32).

---

## Chapter 7: Relevance Filtering & Feature Sampling

### 7.1 Relevance Filtering

```python
async def get_feature_activations_for_prompt(engine, prompt):
    return await engine.get_feature_activations(prompt)
```

#### `get_feature_relevance_scores(feature_tensor, feature_indices)`

```python
def get_feature_relevance_scores(feature_tensor, feature_indices):
    if feature_tensor is None or feature_tensor.numel() == 0:
        return {idx: 0.0 for idx in feature_indices}
    max_activations = feature_tensor.max(dim=0).values
    scores = {}
    for idx in feature_indices:
        if idx < len(max_activations):
            scores[idx] = float(max_activations[idx].item())
        else:
            scores[idx] = 0.0
    return scores
```

**Math:** Uses max aggregation across the sequence dimension to get the peak activation:
$$r_i = \max_{t} A_{t,i}$$

**Input:** Activation tensor $A \in \mathbb{R}^{T \times d_{\text{sae}}}$, list of feature indices.  
**Output:** Dict mapping index $\to$ peak activation (float).

#### `filter_irrelevant_features(relevance_by_feature, max_activation=0.01)`

```python
def filter_irrelevant_features(relevance_by_feature, max_activation=0.01):
    irrelevant_features = []
    for feature_idx, scores in relevance_by_feature.items():
        if all(score <= max_activation for score in scores):
            irrelevant_features.append(feature_idx)
    return irrelevant_features
```

A feature is considered irrelevant if its peak activation is $\leq \epsilon$ on **every** prompt:
$$\mathcal{F}_{\text{irrelevant}} = \left\{ i \;:\; r_i^{(p)} \leq \epsilon \quad \forall p \in \text{prompts} \right\}$$

### 7.2 Feature Sampling

```python
async def sample_filtered_features(
    engine, prompts, feature_labels, n_features,
    concreteness_threshold, num_sae_features,
    candidate_multiplier=3, grader=None, labels_file=None
):
    candidate_pool_size = n_features * candidate_multiplier
    candidate_indices = random.sample(range(num_sae_features), min(candidate_pool_size, num_sae_features))
    
    # Step 2: Concreteness filter
    concrete_features = await filter_concrete_features(...)
    
    # Step 3: Relevance filter
    relevance_scores = await compute_feature_relevance_for_prompts(engine, concrete_features, prompts)
    irrelevant_features = filter_irrelevant_features(relevance_scores, max_activation=0.6)
    
    return irrelevant_features[:n_features]
```

**Algorithm:**
1. Randomly sample $M = \text{candidate\_multiplier} \times N$ candidate indices from $[0, d_{\text{sae}})$.
2. Filter by concreteness: keep only features with real labels scoring $\geq \tau_{\text{conc}}$.
3. Filter by relevance: keep only features with near-zero activation on all prompts.
4. Return up to $N$ features.

**Input:** engine, prompts, labels, $N$, threshold, total features, multiplier, grader.  
**Output:** `List[int]` of selected feature indices.

#### `sample_filtered_features_with_retry()`

If the initial pool yields insufficient features (common when labels are sparse, e.g., only 6.9% of 65536 features are labeled), retries with exponentially increasing pool sizes:

$$\text{multiplier}_{\text{retry}} = 2^{\text{attempt}} \times \text{base\_multiplier}$$

**Input:** Same as above plus `max_retries` (default 3).  
**Output:** `List[int]` (guaranteed to attempt up to 3 retries before giving up).

---

## Chapter 8: Experiment Orchestration

### 8.1 `generate_response()`

```python
async def generate_response(engine, experiment_config, prompt, feature, threshold):
    intervention = None
    if not experiment_config.disable_steering:
        intervention = [{"feature_id": feature.index_in_sae, "value": threshold}]
    convo = [{"role": "user", "content": prompt}]
    seed = random.randint(experiment_config.seed_start,
                          experiment_config.seed_start + experiment_config.n_possible_seeds)
    response = await engine.generate_with_conversation(
        conversation=convo,
        feature_interventions=intervention,
        max_tokens=experiment_config.max_completion_tokens,
        seed=seed,
    )
    return prompt, response, seed
```

**Input:** Engine, config, prompt string, `FeatureInfo`, threshold float.  
**Output:** Tuple `(prompt, response, seed)`.

### 8.2 `get_score_for_prompt()`

```python
async def get_score_for_prompt(engine, judge, prompt, feature, boost, experiment_config, max_retries=3):
    score = None
    retries = 0
    while score is None:
        # Generate steered response
        intervention = None if experiment_config.disable_steering else [{"feature_id": feature.index_in_sae, "value": boost}]
        convo = [{"role": "user", "content": prompt}]
        seed = random.randint(...)
        response = await engine.generate_with_conversation(...)
        score_obj = await judge.grade_response(response, prompt, feature.label)
        
        has_error = "error" in score_obj
        has_attempts = "attempts" in score_obj and bool(score_obj["attempts"])
        if not has_error and not has_attempts:
            has_error = True  # FIX: prevent infinite loop on empty attempts
        
        if has_error:
            retries += 1
            if retries >= max_retries:
                raise Exception(...)
            continue
        score = score_obj["attempts"][0]["score"]
    return score / 100.0
```

**Input:** Engine, judge, prompt, feature, boost value, config, max retries.  
**Output:** Normalized score in $[0, 1]$.

### 8.3 `get_prompt_threshold()`

```python
async def get_prompt_threshold(engine, judge, prompt, feature, experiment_config, show_progress=True):
    threshold = await find_threshold(
        target_score=experiment_config.target_score_normalized,
        get_score_fn=lambda x: get_score_for_prompt(engine, judge, prompt, feature, x, experiment_config),
        prior_mean=experiment_config.threshold_prior_mean,
        prior_std=experiment_config.threshold_prior_std,
        n_trials=experiment_config.threshold_n_trials,
        show_progress=show_progress,
        lower_bound=experiment_config.threshold_lower_bound,
        upper_bound=experiment_config.threshold_upper_bound,
    )
    return float(round(threshold, 2))
```

**Input:** Engine, judge, single prompt, feature, config.  
**Output:** Calibrated threshold float (rounded to 2 decimals).

### 8.4 `get_feature_threshold()`

Finds or retrieves a cached threshold for a feature. Maintains a JSON cache on disk (`threshold_cache_{model_name}.json`) to avoid recomputing thresholds across runs.

**Cache structure:**
```json
{
  "1234": {
    "threshold": 1.25,
    "achieved_score": 0.52,
    "config": { ... }
  }
}
```

**Input:** Engine, judge, feature, all prompts, config.  
**Output:** Threshold float.

### 8.5 `run_one_feature()`

Runs the full pipeline for a single feature:
1. Load prompts.
2. If `per_prompt_calibration=True`: find a separate threshold for each prompt, then generate and grade.
3. Otherwise: find one global threshold (or use precomputed), then generate $N$ responses and grade them all.

**Input:** Engine, judge, config, `FeatureInfo`, optional progress bar, optional precomputed threshold/prompts.  
**Output:** `FeatureResult`.

### 8.6 `run_experiment()`

The top-level orchestrator.

```python
async def run_experiment(experiment_config, sae_filepath, d_model=4096,
                         expansion_factor=16, steering_layer=19,
                         timeout_hours=100, precomputed_features=None,
                         n_prompts_limit=None, output_suffix=None,
                         output_folder=None, repetition_penalty=None,
                         judge_load_in_4bit=True, max_memory=None):
```

**Processing Steps:**
1. **Initialize `HFSteeringEngine`** with explicit `sae_filepath` (fixes bug where SAE path never reached the engine).
2. **Clear CUDA cache** to defragment memory before loading the judge.
3. **Create Judge** early so `ConcretenessGrader` can share its `LocalModelClient` (prevents loading 3 models on 2 GPUs).
4. **Load labels** and sanity-check index ranges.
5. **Sample features** via `sample_filtered_features_with_retry()`.
6. **Drop grader reference** and run `gc.collect()` + `empty_cache()` before the tight generation loop.
7. **Run trials** with an `asyncio.Semaphore` limiting concurrency to `n_simultaneous_features`.
8. **Stream results** to a JSON file after every feature completion (atomic rename from `.tmp`).
9. **Progress logging** via `tqdm` with per-feature status updates.

**Input:** Full config, SAE file path, architecture params, timeout, optional precomputed features, memory caps.  
**Output:** `ExperimentResult`.

---

## Chapter 9: Auxiliary Experiments

### 9.1 Mini Experiment: Single Unrelated Feature

A lightweight diagnostic that:
1. Searches the label CSV for keywords far from generic Q&A (e.g., "recipe", "baseball", "legal").
2. Scans the top 30 matches and picks the one with the **lowest average maximum natural activation** across test prompts.
3. Runs a short Bayesian threshold search (default 5 trials).
4. Generates baseline vs. steered responses side-by-side.
5. Judges the steered responses.

**Natural activation check:**
$$\bar{a}_{\text{max}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \max_t A_{t,i}^{(p)}$$

The feature with the smallest $\bar{a}_{\text{max}}$ is selected as "most irrelevant."

### 9.2 Cooking Feature Experiment

A variant of the mini experiment that specifically filters for cooking/food keywords:
```python
food_keywords = ["recipe", "cooking", "food", "cuisine", "baking", "chef", ...]
```

In addition to finding the threshold and generating responses, this experiment also:
1. Extracts **full per-token activation traces** for the selected feature across all prompts.
2. Tokenizes the prompt to pair each activation with its corresponding token text.
3. Saves a detailed JSON with structure:
```json
{
  "prompt": "...",
  "feature_index_in_sae": 1234,
  "feature_label": "...",
  "sequence_length": 12,
  "max_activation": 0.45,
  "mean_activation": 0.03,
  "tokens": [{"token": "Hello", "activation": 0.001}, ...]
}
```

### 9.3 Multi-Boost Experiment

Instead of finding a single threshold, this experiment sweeps across **multiple boost levels** to characterize the dose-response curve of steering.

#### `compute_boost_levels()`

```python
def compute_boost_levels(threshold_cache, n_levels=10):
    values = np.array([v["threshold"] for v in threshold_cache.values() if v["threshold"] is not None])
    mean = float(np.mean(values))
    std = float(np.std(values))
    levels = np.linspace(mean - 3 * std, mean + 3 * std, n_levels)
    return np.maximum(levels, 0.0), mean, std
```

**Math:** Given a cache of previously found thresholds, computes a linear grid spanning $\mu \pm 3\sigma$:
$$\beta_k = \max\left(0, \; \mu - 3\sigma + k \cdot \frac{6\sigma}{N-1}\right) \quad \text{for } k = 0, \dots, N-1$$

#### `run_one_feature_multi_boost()`

For each feature:
1. Compute baseline threshold (from cache or live).
2. For each boost level $\beta_k$:
   - Generate responses for $N$ sampled prompts.
   - Judge all responses.
3. Store all trials with their corresponding $\beta_k$ in the `threshold` field of each `TrialResult`.

**Input:** Engine, judge, config, feature, boost levels array, progress bar.  
**Output:** `FeatureResult` with `trials` spanning all boost levels.

#### `run_multi_boost_experiment()`

Orchestrator identical in structure to `run_experiment()`, but calls `run_one_feature_multi_boost()` instead of `run_one_feature()`. Saves results to `experiment_multi_boost_{model}_{timestamp}.json`.

### 9.4 Off-Topic Detector (OTD) Discovery

This experiment discovers SAE latents that act as **off-topic detectors**: they fire strongly when the assistant's response is mismatched to the prompt, but stay silent during on-topic conversation.

#### Algorithm

**Step 1: Generate unsteered responses**
$$\mathcal{R} = \{ r_p \mid p \in \mathcal{P} \}$$

**Step 2: Create a derangement**
A derangement $\sigma \in S_n$ is a permutation with no fixed points:
$$\sigma(i) \neq i \quad \forall i$$

This ensures every shuffled (prompt, response) pair is truly off-topic.

**Step 3: Compute activations for shuffled pairs**
For each mismatched pair $(p_i, r_{\sigma(i)})$, construct a conversation and extract max-pooled SAE activations:
$$a_i^{\text{shuffled}} = \max_t \text{SAE}_{\text{encode}}\left(H_t^{(L)}\right) \in \mathbb{R}^{d_{\text{sae}}}$$

**Step 4: Compute activations for normal pairs**
For each matched pair $(p_i, r_i)$:
$$a_i^{\text{normal}} = \max_t \text{SAE}_{\text{encode}}\left(H_t^{(L)}\right) \in \mathbb{R}^{d_{\text{sae}}}$$

**Step 5: Identify detectors**
A latent $j$ is an off-topic detector if:
1. It never fires on normal conversations: $\max_i a_{i,j}^{\text{normal}} < \epsilon$
2. It fires on at least $\tau$ fraction of shuffled conversations: $\frac{1}{N}\sum_i \mathbb{1}[a_{i,j}^{\text{shuffled}} \geq \epsilon] \geq \tau$

Default $\epsilon = 10^{-8}$, $\tau = 0.8$.

**Detector statistics saved:**
- `shuffled_mean_activation`
- `normal_mean_activation`
- `shuffled_active_frequency`
- `normal_active_frequency`
- `activation_ratio`

**Input:** `ExperimentConfig`, activation threshold, off-topic frequency threshold, SAE filepath, memory caps.  
**Output:** `OffTopicDetectorResult` saved to `data/off_topic_detectors_{model}.json`.

---

## Chapter 10: Version Compatibility & Utilities

### 10.1 `to_token_id_list()`

Normalizes the return value of `tokenizer.apply_chat_template()` across `transformers` versions. Some versions return a `tokenizers.Encoding` object instead of `List[int]`, which causes `torch.tensor()` to fail with `RuntimeError: Could not infer dtype`.

**Supported conversions:**
- `torch.Tensor` $\to$ `.tolist()`
- `dict` / `BatchEncoding` $\to$ recursive extraction of `input_ids`
- `tokenizers.Encoding` $\to$ `.ids`
- `List[List[int]]` $\to$ unwrap single batch dimension

**Input:** Arbitrary tokenizer output.  
**Output:** `List[int]`.

### 10.2 `_load_causal_lm_dtype_compat()` and `_load_hf_pipeline_dtype_compat()`

Newer `transformers` renamed the `torch_dtype=` argument to `dtype=`. These wrappers try `dtype=` first and fall back to `torch_dtype=` on `TypeError`, making the notebook robust to version drift.

### 10.3 Key Bug Fixes Documented in Source

1. **SAE path bug:** `run_experiment()` now takes `sae_filepath` explicitly instead of relying on a broken registry lookup.
2. **total_sae_features bug:** CSV labels lack metadata, so `total_sae_features` must be set to $d_{\text{model}} \times \text{expansion\_factor}$ or sampling draws from the wrong range.
3. **0-features retry bug:** `run_experiment()` now calls `sample_filtered_features_with_retry()` instead of the non-retrying version.
4. **Tokenizers.Encoding bug:** `to_token_id_list()` handles native `Encoding` objects.
5. **dtype rename bug:** Compatibility wrappers handle `torch_dtype` vs `dtype`.
6. **Dead shell magic:** Removed `!PYTORCH_ALLOC_CONF=...` which ran in a throwaway subshell; the env var is now set via `os.environ` before CUDA init.
7. **Judge client sharing:** `ConcretenessGrader` accepts `local_client` to avoid loading a third 8B model.
8. **Infinite retry loop:** `get_score_for_prompt()` treats empty `{"attempts": []}` as retryable, not just explicit errors.

---

## Appendix: Symbol Reference

| Symbol | Meaning |
|--------|---------|
| $d_{\text{model}}$ | Hidden dimension of the base LLM (4096 for Llama-3.1-8B) |
| $d_{\text{sae}}$ | SAE dictionary size = $d_{\text{model}} \times \text{expansion\_factor}$ (65536) |
| $L$ | Target transformer layer for steering (19 in this notebook) |
| $v$ | Steering boost strength (scalar multiplier) |
| $v^*$ | Calibrated threshold found by Bayesian search |
| $h$ | SAE feature activation vector |
| $d_i$ | Decoder direction for feature $i$ |
| $\epsilon$ | Activation threshold for relevance/OTD filtering |
| $\tau$ | Minimum frequency threshold (e.g., 0.8 for OTD) |
| $s$ | Judge score in $[0, 1]$ |
| $w$ | Bayesian update weight (default 5) |
| $\mu_{\text{prior}}, \sigma_{\text{prior}}$ | Gaussian prior parameters for threshold search |

---

*End of Document*

---

**To convert this to PDF:** Save the above content as `reference.md` and run:
```bash
pandoc reference.md -o Endogenous_Steering_Resistance_Reference.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

Or use any Markdown editor with PDF export (Typora, VS Code Markdown PDF, etc.). All math is standard LaTeX and will render correctly.