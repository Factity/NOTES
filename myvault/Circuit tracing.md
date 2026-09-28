
Sparse autoencoder models or transcoder models or simply cross coder models have emerged as some of the promising alternatives for identifying interpretable features represented in superposition. These methods as we have seen in the above sections are useful to decompose models into sparsely active components. While the current sparse encoding mechanisms are promising mapping the activations and correlating these with human understandable concepts is still very hard and cumbersome. Although they are cumbersome that is the best we got that could possibly applied at wider scale. 








key takeaways and elements

Transcoders: extract features using transcoders rather than SAEs. which can be used as a replacement models. These can be used to study feature interactions directly in MLP blocks. These can be of cross layer as in every subsequent layers have connections to the cross layer transcoder from the previous mlp block. We can essentially train a cross layer transcoder by replacing an MLP block in one of the sections. Attribution graphs are used to explain the steps a models input tokens goes through while being mapped to the output token. The neural path The nodes in the attribution graphs consists of token embeddings of the prompt, activations of the features, reconstruction errors and logits. 

Linear attribution between the features - each interaction between the features is linear. feature interactions are often indirect and happen mostly using the attention patterns and normalization blocks. Even though the structure due to these are sparse there are simply too many features active on a prompt by identifying the nodes we can prune them to identify the ones that most contribute to the output. 

Evaluation- How do we ensure what we have designed is the rightfully maps the MLP activations to the cross layer transcoders. By applying the different forms of prompt perturbations to the inputs if the observed deviations are indeed validate our results. This can be done observing the changes in the feature directions as the prompts are varied and observing the feature activations. Naive global weights are often less interpretable than attribution graphs due to weights interference. 

Cross layer transcoder 

The cross layer transcoder consists of L layers the same as the number of MLP blocks in the transformer and each of the layers are connected to the residual stream and each of the upcoming n mlp layers. where n is the current layer number. Each of the features are read from the residual stream followed by a jump relu non linearity. 

if n is the current layer the $a^n$ = $JUMPRELU(W_{enc}^nx^n)$ here $x^n$ represent the original models weights from the residual stream and $W_{enc}^n$ is the weights used for encoding in the nth layer. The output consists of the $y^n$ = $\sum_{n^`=1}^n (W_{dec}^{n\to n^`} a^{l^`})$  where  $y^n$ is the  output of the original model specifically  and dec here refer to the decoder weight matrix. 

To train these models we need to focus these two most important aspects trying to match as much as possible of the original models MLP block and the sparsity penalty 

for this the loss with respect to the MLP mismatch is give by $L_{MSE} = \sum_{n=1}^L||y^n - \hat{y}^n  ||$ here $y^n$ is the loss from the MLP  block and other is the output from the transcoder. 

The second score we need to keep track of is the sparsity penalty here there are two important hyperparameters. $\lambda$ and $c$ summed across layers. 

