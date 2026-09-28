


RLHF has emerged to be one of the most important training technique in post training. Aligning the model is of atmost important this is possibly the closest we have gotten in terms of providing the models with human interpretable goals. Definition of a goal is in and itself is a very complex task with the real world nuances involved. RLHF is the most promising we have ever gotten to aligning the models. In the spirit of that I have tried to document what I have learnt in terms of aligning the models the quirks associated with using the techniques etc in the following blog post. There are several techniques with far reaching implications across various fields. I tried to put togther a couple of ways which are promising. 

I highly --------> link to the book 

RlHf emerged as a consequence of embedding human information into AI systems. As a method to solve the hard to specify problems. inexpessibility of the nature of the preferences. Can we solve it with only the basic preference signals guiding the optimization process. 

Three step process of the RLHF 

pipeline of RlHF 

1.  Model must be trained 
2. human preference data must be collected for the training of a reward model of human preferences 
3. Optimized with the RL optimizer sampling and rating them 


Some of the early reinforcement learning techniques were applied across 

-  Deep reinforcement learning 
- Summarization 
- following instructions 
- parsing web information for question answering 
- alignment 

Post training is the process by which language models are made useful for  downstream tasks. 

Post training can be thought of as a many stage training process which can be broadly classified into these 3 main types 
1. Instruction / Supervised Fine tuning - Learning features in the language 
2. Preference Fine tuning 
3. Reinforcement learning with verifiable reward 


For the ease of understanding the notation I have added the following taxonomy 
If some pof the examples doesnt make sure Please dont worry each of the method presented here will be explained throughly in the later sections 

> **Symbols and their origins:**
> - $x \in \mathcal{X}$: a prompt or input. Drawn from a prompt distribution $\mathcal{D}$.
Example - what is the capital of France ?

> - $y \in \mathcal{Y}$: a response or completion. Can be a sequence of tokens $y=(y_1,\dots,y_T)$.

Example - The capital of france is paris. 

> - $\pi_\theta(y\mid x)$: the policy, a parametric model (e.g., a transformer) that outputs a probability distribution over responses given a prompt. $\theta$ are the parameters.
  
  **Example:** For $x =$ `"What is the capital of France?"`, the model might output:
  $$
  \pi_\theta(\text{"The capital of France is Paris."} \mid x) = 0.42
  $$
  $$
  \pi_\theta(\text{"Paris."} \mid x) = 0.31
  $$
  $$
  \pi_\theta(\text{"London."} \mid x) = 0.05
  $$
  It assigns a probability to every possible response.
  

> - $\mathcal{D}$: the distribution over prompts. Often a fixed dataset of prompts, but can be a broader distribution.
  **Example:** A dataset of 10,000 instruction prompts:
  ```json
  [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "Write a Python function to add two numbers.",
    ...
  ]
  
  ```

> - $f$: a feedback signal. Its nature differs per method: a demonstration, a preference pair, or a verifier score.

  **Examples:**
  - **SFT:** $f =$ `"The capital of France is Paris."` (a demonstration)
  - **Preference FT:** $f = (\text{"The capital of France is Paris."}, \text{"The capital of France is London."})$ where the first is preferred.
  - **RLVR:** $f = V(x,y) = 1$ if $y$ contains `"Paris"`, else $0$ (a verifier score).


> - $\mu_\theta(y\mid x)$: the distribution from which responses are sampled during training. It can be the data distribution (off-policy) or the policy itself (on-policy).

  **Examples:**
  - **SFT / DPO (off-policy):** $\mu_\theta(y\mid x) = \mathcal{D}_{\text{SFT}}(y\mid x)$. We sample $y$ from the fixed dataset, e.g., always `"The capital of France is Paris."`
  - **RLHF / RLVR (on-policy):** $\mu_\theta(y\mid x) = \pi_\theta(y\mid x)$. We sample $y$ from the model itself, e.g., the model might generate `"Paris is the capital."` or `"The capital is Paris."` depending on its current parameters.


> - $\ell(\theta; x, y)$: a loss function measuring how poorly the policy matches the feedback for a given prompt and response.

  **Examples:**
  - **SFT:** $\ell = -\log \pi_\theta(\text{"The capital of France is Paris."} \mid x)$. If the model assigns probability $0.42$ to that full response, then $\ell = -\log(0.42) \approx 0.87$.
  - **Preference FT (DPO):** $\ell = -\log \sigma\bigl(s_\theta(x, y^+) - s_\theta(x, y^-)\bigr)$. If $s_\theta(y^+) = 2.1$ and $s_\theta(y^-) = 0.3$, then $\ell = -\log \sigma(1.8) \approx 0.17$.
  - **RLVR:** $\ell = -V(x,y)$. If the verifier returns $1$ for a correct answer, $\ell = -1$.

> - $\Omega(\theta)$: a regularizer, typically a KL divergence to a reference policy $\pi_{\text{ref}}$. It prevents the policy from drifting too far, preserving fluency and preventing reward hacking.


  **Example:** For a prompt $x$, the KL divergence between the current policy and the reference policy is:
  $$
  \Omega(\theta) = D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x) \,\|\, \pi_{\text{ref}}(\cdot\mid x)\bigr)
  $$
  If $\pi_\theta$ and $\pi_{\text{ref}}$ are identical, $\Omega = 0$. If $\pi_\theta$ puts more probability on `"Paris"` and less on other tokens, $\Omega > 0$.


> - $\beta \ge 0$: a scalar controlling the strength of the regularizer.

  **Example:**
  - $\beta = 0.01$: weak regularization, the policy can drift far from the reference.
  - $\beta = 0.1$: moderate regularization.
  - $\beta = 1.0$: strong regularization, the policy stays very close to the reference.




Pre training is the pure Maximum likelihood estimation problem on the raw text. There is no prompt- response distinction, no feedback signal, no refrence model regularizer. 


$$
\theta_{\text{pre}} = \arg\min_\theta -\mathbb{E}_{x \sim \text{raw text}} \left[ \sum_{t} \log \pi_\theta(x_t \mid x_{<t}) \right]
$$

here were are sampling from some raw text, each of which is before a certain index token t and the responses are produced using the trained policy 



Cite the link for the maximum likelihood estimator 
https://www.youtube.com/watch?v=rCdxlN6Ph14



The objective of the post training is to align the pretrained base model with human preferences 
With the above notation it can also be represented by 


> **General objective:**
> $$
 \theta^*
 =
 \arg\min_\theta
 \;
 \mathbb{E}_{x\sim\mathcal{D}}
 \;
 \mathbb{E}_{y\sim \mu_\theta(\cdot\mid x)}
 \left[
 \ell(\theta; x, y)
 \right]
 +
 \beta\,\Omega(\theta)
 $$
> 
> The key differences between the above mentioned 3 methods arise as a consequence of  difference in :
> 1. What feedback $f$ is used.
> 2. Whether responses $y$ are sampled from data or from the policy.
> 3. What loss $\ell$ is applied.
> 4. Whether a regularizer $\Omega$ is used.



Now the general objective of all the 3 different methods is defined above let us look at the all 3 different associated methods quirks associated with each of them and the safety methods built on top of each of them. 



## Supervised finetuning




















Contrastive loss function: Loss which is computed from the completion between two or more examples. Instead of training the model on the specific examples we show it which ones are better and the model learns from it. 

Some of the implementation challenges fro RLHF - 

-  how to control the optimization 
-  Optimization process itself is prone to over optimization 
- RLHF needs a strong starting point it is simply not the end all be all but it works 

Costlier than simple fine tuning some of the challenges include legth bias absolute performance matters a lot. 


Brief overview of the RLHF recipe 


next-token prediction loss function on a set of carefully crafted datapoints where the model is shown only data in this question-answering format.


-  training a reward model  that actually captures the human preferences
- A question answering model 

So question answer model does the completions reward model ranks them and then we use RL to make the reponses better 

Once the performance is saturated the final model is served 


Extracting a to of performance from simple static base model 

Mid training annealing 


The adept example is the math problems no matter how many times you eread the theory it simply doesnt work unless you do worked examples. It simply learns the nuances of pattern recognition 

Less is more for alignment 

improve a narrow set of evaluations, such as AlpacaEval, MT-Bench, Arena (formerly Chatbot Arena, a platform



















