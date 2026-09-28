pgradeI have been an avid advocate of interpretability, but the current climate does not seem to be on my side. Although it is the right thing to do, economics doesn't seem to be on my side either. Moving from pure math to industry, I often look at things from a mathematician's perspective which is sadly too-idealistic. My first instinct was to think about existence (∃), but not about whether it is tractable. Because I was reading at my usual fast speed, I skipped over the phrase “in the short term.” But, playing devil's advocate as I should, I will present both sides of the argument.



Some rough notes to edit the post around it 

The author talks about the promise of using latent space monitoring 

what is latent space monitoring you might ask ? 


![[Pasted image 20260902155630.png]]


These defenses act as scanners that seek to detect harmful activations before they lead to undesirable actions 

can they really hide stuff ? 

what are the methods they took 

sparse autoencoders
representation probing 
latent OOD detection 

the above mentioned all methods are susceptible

 harmful behavior via inconspicuous latent
states?

against probes trained to classify harmfulness,
our attacks can often reduce recall from 100% to 0% while retaining a 90% jailbreaking rate

how robust are these latent space defenses ?

The author talks about if they try to bypass latent space monitors they have 2 different types of goals for this -
-  they should have the ability to induce inconspicuous activations 
- those activations should lead to the desired behavior by the attacker. 

So if the latent space has only single harmfulness direction and the activations that align with the direction produce harmful outputs. While also fooling linear classifier inducing this direction.  

They found that by jail breaking obfuscation attacks can bypass wide monitors.

The author used different attack vectors like
-  adversarial suffixes
-  data poisoning 
- model fine tuning 

The author talks about the difficulty of telling the difference between inconspicuous and harmful activations they have found benign activation clusters having obfuscated activations close to it. 

To test it in another way they have trained a defender on examples of obfuscated activations if there were only few of them the monitor should quickly be able to cover them all and become robust. 

They cannot induce arbitrarily activation combinations. For let's say an SQL detection probes, the model still generates SQL but its code is incorrect. 

Some of the deeper questions that might arise here:

let's say if there is a unique characteristic in the response which directly correlates with the performance drop we might be able to say that the an obfuscated input might be present. 

They have used multiple types of detectors to show that indeed in those ones these is characteristic obfuscation - 

Harmfulness monitoring 

Linear, MLP probes fails to flag when ingested with obfuscation attacks either with hard or soft prompts 

Failure in models produced with latent robustness training methods, latent adveserail training methods 

Backdoor detection 

full control of the model training process 

even when the access is not full using the data poisioning by choosing a complex trigger or low precision rate 




setting - 
jail breaks 
back doors 
sql injection 
image advexes 

attack vectors - 
hard prompts 
embeddings 
fine tuning 
poison the data 

Defend - 
probe it 
find the ood one out anomoly detection 
sparse autoencoders (linear transform to bigger space to isolate concepts )
circuit breakers 


Training models to easily monitor against these attacks 










link to the training experiment for the obfuscation attacks  



https://arxiv.org/pdf/2412.09565

https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/llama3-jailbreaks

https://github.com/LukeBailey181/obfuscated-activations/tree/main



Once this replication is done 



you need to prove that activations although intractable does indeed contain the information 


this is not formally published but but but 


Thinking of this thought experiment:


let us imagine a path on the graph with the activations 

only connected sequences are gathered if disconnected the activation does not map 

a hypothetical activation snake that jumps from node to node and the snake turns red when it is on an obfuscated node and turns green when it is not on one 


if we were to find two snakes which stays entirely red or entirely green throughout the traversal and no non activation nodes have snake connected to it we can separate an obfuscated node from a normal one 

if the above could be done we can theoretically prove that there exists some activation which led to the right response 

any activation that doesn't directly materialize in the response generation could be eliminated.












paper under discussion 


https://arxiv.org/pdf/2412.09565       




Why interpretability is the right way to do this ?


old interpretability 


file:///home/mukullight/Downloads/price_invert.pdf



cost estimator


https://ai-cost-estimator.com/blog/anthropic-j-space-global-workspace-claude-interpretability-safety-cost

The economics of the argument ?

https://www.longtermwiki.com/wiki/E297

modeling with the security investmnent 

https://arxiv.org/pdf/2604.23058



implementataion costs integration costs 

https://openreview.net/pdf?id=BpyR6Wkl1c



The pessimistic engineer argument 

I am not sure if could even call it pessimistic may be they are realists and the researchers are the naive optimists 



The optimistic researcher argument 



What I think 








Some stupid stuff you need to delete before you publish 

Should you do the replication or should you just write the results ? 







One of the key things to note is once you pick a paper or an experiment dont leave it unfinished finish it before you wake up context switching is expensive and keeping the idea in the head is not plausible jot it on the paper as soon as possible. 

AIm for a 4 hour time slot 

-  1 hour reading the paper and the code 
- 1 and half hour of writing it down 
-  1 hour for code setup and replication 

half an hour for th buffer if an absolute necssity shows up 
