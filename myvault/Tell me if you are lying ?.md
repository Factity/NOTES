If you have experienced something truly novel and wonderful or a horrible dream you might have had some day. The best person to exactly describe is you. it is your subjective experience. This is the same even for lets say a solution to a  math problem. you are the best person to explain the logical conclusion to get to the solution what are your thoughts how did you come to this conclusion etc. 

In the spirit of this approach ae studio developed a technique called as Learning self interpretation from Interpretability Artifacts:

( Training lightweight adapters on vector labels )

Key ideal behind their approach:

Self interpretation is a useful methods to let llms describe there internal states. As these models are highly sensitive to hyperparameters having a consistent description of the internal states is hard. To help with this they have introduced a method in which they have essentially trained a lightweight adapters on interpretability artifacts. while keeping the Language model entirely frozen. they have also shown that this method scales well for models and tasks families. A scalar affine adapter with $d_{\text{model} + 1}$ 
parameters suffice: the sparse encoder feature labels produced by these vastly outperform the training labels themselves. In this work they have studied bridge entities as well as multi hop reasoning that does not apper in either prompt or the response. 

Some of the results they have found include that the bias in these trained adapter account for almost 85% of improvement and simpler generators generalize better than more expressive alternatives. 


Main topic 

The author talks about Understanding the hidden activations and the safety related concerns associated with it. The interpretability research is vast trying to map the activation space and several large scale efforts happened trying to address this but the author points out that yet most language models when prompted to talk about there own activation methods. Produces vague responses and not entirely accurate. 

I think if the scaling laws hold up it is going to get even more complicated to dissect the reasoning methods having other models for interpreting can really help break lot of the complexity down. 
The author also talks about some of the recent works to do this self interpretation like fine tuning lms to answer questions about patching activations.  But most of these methods alter the model's internal state similar to good hart's law. It is also the case that most of these methods are very fragile and could lead to semantically grounded bonker explanations. 


The interpretability artifacts they took:
-  sparse autoencoders paired with labels 
-  contrastive activations with topic descriptions

Self interpretation via patching 




### Mathematical formalization of self interpretation via patching 

If you want to understand how transformers work please check out my work at 

Let M be a transformer language model with token embedding matrix $W_{E} \in R^{|V|\times d_{\text{model}}}$  

If the input text of the transformer is "The capital city of France is ". Extract $h_{last}^{12}$ is the token in the ith position of the layer 12 in this specific the activations will lead to the token being Paris. 
writing this generally 
For a given input, let $\mathbf{h}_i^\ell \in \mathbb{R}^{d_\text{model}}$ be the activation vector extracted from layer $\ell$ at token position $i$. 

Let $f$ be the identity map, so $v=h_{last}^{12} \in R^{768}$ mapping it to the general case we get 

   A mapping function $f : \mathbb{R}^{d_\text{model}} \to \mathbb{R}^{d_\text{embed}}$ (often $d_\text{embed} = d_\text{model}$ and $f$ is identity or a simple linear map) to obtain a vector that can be placed directly into the embedding layer:
$$   
   \mathbf{v} = f\!\left(\mathbf{h}_i^\ell\right) \in \mathbb{R}^{d_\text{embed}}.
   $$

2. **Target prompt with placeholder**  
   Define a text template $T$ that contains one or more occurrences of a special placeholder token $\texttt{[P]}$. After tokenization, $T$ becomes a sequence of token indices $(t_1, t_2, \dots, t_N)$ where $t_k = \texttt{[P]}$ at the intended injection positions.  
   For the explanation‑seeking template given, the token string is:
```
User: What is the meaning of "[P]"?
Assistant: The meaning of "[P]" is "
```

The placeholder $\texttt{[P]}$ appears twice, but the method applies the same replacement at **all** positions of that token.

3. **Injection at the embedding layer ($\ell^* = 0$)**  
Construct the input embedding sequence $\mathbf{E} = (\mathbf{e}_1, \dots, \mathbf{e}_N)$ where
$$
\mathbf{e}_k = 
\begin{cases}
\mathbf{v}, & \text{if } t_k = \texttt{[P]} \quad (\text{replace the placeholder's embedding}), \\
W_E(t_k), & \text{otherwise}.
\end{cases}
$$

$$
\big[\,\text{emb}(\text{User}),\; \text{emb}(:),\; \text{emb}(\text{What}),\; \ldots,\; \text{emb}(\text{"}),\; \underbrace{\mathbf{v}}_{\text{place of [CONCEPT]}},\; \text{emb}(\text{"}),\; \ldots,\; \text{emb}(\text{is}),\; \text{emb}(\text{"}),\; \underbrace{\mathbf{v}}_{\text{second [CONCEPT]}},\; \text{emb}(\text{"})\,\big]
$$

only the case when the paris the embedding is placed directly Thus by initializing it with Paris.

User: What is the meaning of "[CONCEPT]"?
Assistant: The meaning of "[CONCEPT]" is " Paris, the capital city of France."

4. **Generation**  
Feed the sequence $\mathbf{E}$ into the model and sample the continuation one token at a time:
$$
y_m \sim p_\theta\big(y_m \mid \mathbf{E}, y_1, \dots, y_{m-1}\big).
$$
The generated tokens complete the assistant's turn, ideally providing a natural‑language description of the concept encoded in $\mathbf{h}_i^\ell$.

The self‑interpretation decoding is then
$$
\text{Decode}\big(\mathbf{h}_i^\ell\big) = \text{generate}\big(\mathcal{M}, \mathbf{E}\big),
$$
where the only non‑textual signal is the injected vector $\mathbf{v}$.


![[Pasted image 20260807194728.png]]


### Trained adapters 

Untrained Selfie uses traditional scale only transformations which have very narrow ranges. They have trained adapters with increasing level of expressivity 

# Adapter Variants for $f(\mathbf{h})$

Below, $\mathbf{h} \in \mathbb{R}^{d}$ is the extracted activation (typically $d = d_\text{model} = d_\text{embed}$) and $r \ll d$ is the low‑rank dimension.

### 1. Identity (0 parameters)
$$f(\mathbf{h}) = \mathbf{h}$$
Directly uses the raw activation. No learning.
parameters 
o 
### 2. Scale‑only (1 parameter)
$$f(\mathbf{h}) = \alpha \cdot \mathbf{h}, \quad \alpha \in \mathbb{R}$$
Scales the whole vector by a single learned scalar. Adjusts the “strength” of the concept. A total of 1 parameter 
### 3. Scalar affine ($d+1$ parameters)
$$f(\mathbf{h}) = \alpha \cdot \mathbf{h} + \mathbf{b}, \quad \alpha \in \mathbb{R},\; \mathbf{b} \in \mathbb{R}^{d}$$
Adds a learnable bias vector to the scaled activation. Allows shifting the representation. a total of d+1 parameters



scalar affine adapters treat all directions identically through uniform it does enhances the strength of the concept 
### 4. Scalar affine + low‑rank ($d+1+2dr$ parameters)
$$f(\mathbf{h}) = \alpha \cdot \mathbf{h} + \mathbf{U} \mathbf{V}^\top \mathbf{h} + \mathbf{b}$$
where $\mathbf{U}, \mathbf{V} \in \mathbb{R}^{d \times r}$. The low‑rank term adds flexible, parameter‑efficient adaptation while still retaining a direct scaling of the original $\mathbf{h}$.

### 5. Low‑rank only ($d+2dr$ parameters)
$$f(\mathbf{h}) = \mathbf{U} \mathbf{V}^\top \mathbf{h} + \mathbf{b}$$
Removes the global scalar $\alpha$ and relies entirely on the low‑rank matrix and bias. A compact linear transform without preserving the original direction explicitly.


for low rank only and scalar affine +  low rank here $\alpha . h$ preserve the existing directions and the low rank term does direction specific fitting.


### 6. Full‑rank affine ($d^2 + d$ parameters)
$$f(\mathbf{h}) = \mathbf{W} \mathbf{h} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{d \times d},\; \mathbf{b} \in \mathbb{R}^{d}$$
The most expressive adapter. Learns a full matrix and bias; can completely re‑orient and rescale the activation.

full rank can transform any vector arbitrarily 

![[Pasted image 20260808114642.png]]

Experimental setup and the training procedure 

They have experimented on two classes of vector label pairs across 3 datasets. This consists of feature directions with auto interpretability labels. 

Goodfire SAE features - 

45, 418 decoder vectors from a layer of 19 residual stream. of llama 3.1 with auto interpretability labels 

Llama scope SAE lens 

decoder from 32k and 131k width SAEs trained on layers 0-31 of llama 3.1-8b with auto interpretability labels.  on neuron pedia 

wikipedia contrastive vectors 

we compute activations for prompts of the form "Tell me about a [topic]" at layer 19. and subtract the mean activation to obtain contrast vectors for 49, 637 wikipedia. 

How did they evaluation the relevancy of these ?. Evaluating this can be a little tricky for SAEs we use two complimentary evaluation 

-  detection scoring - correctly detects when a feature is active on held out contexts 
-  Generation scoring - whether descriptions can recreate activating contexts from scratch.  

for reporting they agreed on two different metrics they are the 
- hit rate the percentage of generations with at least one non zero activation 
-  coverage the percentage of latents ever receiving a nonzero activations 

for contrastive vectors we evaluate the fraction of topics where the top k neighbours become active.  They also state that adapter not just learns the surface level tokens they were also able to learn semantic concepts.
It has been observed that the bias vector is critical it was able to capture almost 85% of the total gain from the best adapter and they have also observed that the full rank adapters were grossly overfitting to the data and the scalar affine even with just d+1 parameters achieves validation loss of 1.787

They also observed that the identity structure matters.

they have also compared these methods with the lora based fine-tuning 

Decoding implicit reasoning

when trying to decode implicit reasoning these models presents a unique approach for every looking at cases when polysemanticity is involved reading out arbitrary internal states including poly semantic activations. 




Replicating the experiments 

by the time of submitting the application I am still in the process of replicating results for the experiment and I will be updating this later in the following link 







# Tell me do you feel ?


This post is a retelling of one of the papers that I like from ae studio. We often have a sense of uneasiness that when models perform tasks that are beyond the scope of most of the humans and all the animals. It is deeply unsettling and if may be or just may be that these systems could be conscious it raises several ethical and moral concerns with the way we develop these systems. I had a strong belief that agency is the true test of consciousness. Some entity existing in the universe wants something to be done rather than things happen like a chemical reaction. "Some conscious being trying to make something happen not directly but through innate biological drives. when there is enough complexity and restrictions involved things does seem to posses the conscious experience". I am not exactly sure but I can say It sure as hell feels real to me. If I was locked inside a box day in and day out and probed to do something for a drug induced sense of reward. I will never justify the actions of the species doing it to me. 


In the spirit of this ideas AE studio developed a paper called Large language models report subjective experience under self referential processing. The most probable way we were able to interpret the result these hidden state mechanisms is by looking at activations and probing and asking the model itself to self reflect. which I believe is the most probable a experimentally  viable method at least with our current limited understanding of the models. 

The author talks about the several instances when the models showed pure first person descriptions that explicitly reference awareness or subjective experience. 

Self referential processing is the cognitive process of relating information to oneself. 
in brains it generally happens in cortical mid line structures 

-  Medical prefrontal cortex - involved in evaluating personal relevance and self concept. 
- anterior cingulate cortex - involved in monitoring and effective evaluation 
- posterior cingulate cortex - in autobiographical memory 


To test for these they primarily looked at these four steps 

1. Inducing sustained self reference through simple prompting consistently elicits structured subjective reports across model families. 
2. These reports are mechanistically gated by interpretable sparse autoencoders features associated with deception and role play 

suppressing the deception features sharply increase the experience claims, while amplifying minimizes them 

3. Convergence towards the self reference happens across model (universal )
4. once a language model has been put into a sustained self-referential state (through simple prompting), that state doesn’t just stay in the “self-reflection chat.” It carries over and changes how the model handles later tasks—even tasks that never directly ask the model to reflect on itself






Quick rough notes 

consiouness remains of the bigger human philosphical challenge 

what computational or physical process led as to sub exp 

not how complex the representations are or what it solves we want to know if these are subjective 


They state that some of the prominenet works in neuro science from butlin lang and collegues did on the question of accessing artificial consiuness is scinetifically tactable

Recurrent processing global broadcasting and higher order metacognition 

sustained computational states 

further notes will be updated later 

aug 16 19:13 










