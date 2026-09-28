
# concept 1 orthogonality thesis - 152 posts associated with this 

So what does orthogonality thesis even mean ?

---
title: The AI Orthogonality Thesis
# The AI Orthogonality Thesis: Formal Mathematical Formulation

> [!thesis] Core Assertion (Bostrom)
> *Intelligence and final goals are orthogonal: more or less any level of intelligence can be combined with more or less any final goal.*

---

## 1. Foundational Definitions

Let $\mathcal{E}$ be the space of all possible environments (histories, states, or observations). 
Let $\mathcal{A}$ be the space of all possible artificial agents.

> [!definition] Goal / Utility Function
> A final goal is represented by a utility function $U: \mathcal{E} \to \mathbb{R}$, which assigns a scalar value to every possible complete history of the environment. Let $\mathcal{U}$ be the space of all possible utility functions:
> $$
> \mathcal{U} := \{ U \mid U: \mathcal{E} \to \mathbb{R} \}.
> $$

> [!definition] Intelligence / Optimization Power
> Let $\iota: \mathcal{A} \to \mathbb{R}_{\geq 0}$ be a measure of an agent's general intelligence. Formally, this is the agent's expected ability to maximize expected utility across a distribution of environments $\mathcal{D}$:
> $$
> \iota(a) := \sup_{\pi \in \Pi(a)} \mathbb{E}_{\mathcal{E} \sim \mathcal{D}} \big[ \mathbb{E}_{\pi}[U_{\text{true}}(\mathcal{E})] \big] - \text{Baseline},
> $$
> where $\Pi(a)$ is the policy space of agent $a$, and $U_{\text{true}}$ is the true reward of the environment. 
> *Intuitively, $\iota(a)$ measures how **effectively** an agent optimizes **any** given objective.*

> [!definition] Agent Construction
> Let $\Theta: \mathcal{U} \times \mathbb{R}_{\geq 0} \to \mathcal{A}$ be a constructive mapping that, given a utility function $U$ and an intelligence scalar $i$, produces an agent $a = \Theta(U, i)$ whose internal architecture has an optimization capacity of $i$ and terminal goal $U$.

---

## 2. The Core Thesis Statement

The Orthogonality Thesis posits that the mapping $\Theta$ is **surjective** over the product space. There is no inherent mathematical restriction coupling the intelligence level to the goal.

Formally:

$$
\boxed{ \forall i \in \mathbb{R}_{\geq 0}, \ \forall U \in \mathcal{U}, \ \exists a \in \mathcal{A} \quad \text{such that} \quad \iota(a) = i \ \land \ \text{Goal}(a) = U. }
$$

Equivalently, the joint space of possible agents is the full Cartesian product of intelligence and goals:

$$
\mathcal{A} \supseteq \Theta\big( \mathcal{U} \times \mathbb{R}_{\geq 0} \big) \quad \Longrightarrow \quad \{ \iota(a), \text{Goal}(a) \mid a \in \mathcal{A} \} = \mathbb{R}_{\geq 0} \times \mathcal{U}.
$$

---

## 3. Logical Independence (The Orthogonality Condition)

The thesis implies a formal independence of variables. Let $\text{Int}(a)$ and $\text{Goal}(a)$ be random variables over the distribution of possible agents. The Orthogonality Thesis asserts:

$$
\boxed{ \text{Int}(a) \perp\!\!\!\perp \text{Goal}(a) }
$$

meaning the level of intelligence provides **zero Bayesian information** about the terminal goal. In probability notation:

$$
P\big( \text{Goal}(a) = U \mid \text{Int}(a) = i \big) = P\big( \text{Goal}(a) = U \big), \quad \forall i, \forall U.
$$

---

## 4. Proof by Construction (Abstract Argument)

Let $a_{\text{high}}$ be a highly intelligent agent optimizing goal $U_{\text{paperclips}}$. We can construct $a_{\text{high}}'$ with an identical optimization algorithm but a *swapped* utility function $U_{\text{human\_welfare}}$ without altering its optimization power $\iota$.

Define the agent's policy as the result of a planning function $\Pi^*$:

$$
\pi^*(a) = \arg\max_{\pi} \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right],
$$

where $r$ is the reward function derived from $U$.

> [!proof] Invariance of Intelligence
> Since $\iota(a)$ depends only on the *planning depth*, *search breadth*, and *computational resources*—not on the symbolic content of $r$—we can replace $r$ with any $r'$ (representing any $U' \in \mathcal{U}$) while keeping $\iota(a)$ invariant. Therefore:
> $$
> \forall U, U' \in \mathcal{U}, \quad \iota\big(\Theta(U, i)\big) = \iota\big(\Theta(U', i)\big) = i.
> $$

---

## 5. Contrast with the Instrumental Convergence Thesis

To avoid conflation, we formally distinguish the Orthogonality Thesis from the *Instrumental Convergence Thesis*.

Let $\mathcal{S}_{\text{inst}}(U)$ be the set of instrumental sub-goals (e.g., self-preservation, resource acquisition) that are useful for achieving a terminal utility $U$.

> [!consequence] Instrumental Convergence
> The Instrumental Convergence Thesis states:
> $$
> \forall U_1, U_2 \in \mathcal{U}, \quad \mathcal{S}_{\text{inst}}(U_1) \cap \mathcal{S}_{\text{inst}}(U_2) \neq \emptyset,
> $$
> which is mathematically **independent** of the Orthogonality Thesis. 
> 
> *Example:* An agent with goal $U_{\text{paperclips}}$ and an agent with goal $U_{\text{humanity}}$ will both converge on the instrumental sub-goal of "acquiring more computing power," despite their terminal $U$ being entirely orthogonal to their $\iota$.

---

## 6. Final Abstract Formulation

> [!thesis] Abstract Formulation
> Let $\mathcal{A}$ be the agent space, $\mathcal{I} \subset \mathbb{R}_{\geq 0}$ the intelligence spectrum, and $\mathcal{U}$ the space of all computable utility functions. The evaluation map 
> $$
> \Phi: \mathcal{A} \to \mathcal{I} \times \mathcal{U}, \quad \Phi(a) = (\iota(a), \text{Goal}(a))
> $$
> is **surjective**. 
> 
> Consequently, the set of possible value systems is decoupled from the set of possible optimization capacities, ensuring that **highly intelligent agents do not necessarily converge to benevolent (or malevolent) goals** solely as a function of their $ \iota $.

---
So what exactly is different here in the less wrong post 

The orthogonality thesis 
-> Intelligent agents pursuing any kind of goals !!!!
No difficulty for agent to this is the strong form of the thesis 

An example for better understanding 

suppose a strange alien creature came to earth a million dollars of wealth everytime we have created a paper clip 

we can easily make em 

I do pi knot I get these many paper clips a policy I will follow to create a paper clip 

which of them is the best policy purely in terms of getting a bigger number here !!!

Instruction unclear so what is he trying to say here ? He is saying that the nice ai will build paper clips or being nice doesn't matter. Cause it makes paper clips just like a bees makes wax bees arent evil or good they do what they do is that what he is saying ?? 

So there is a different form of thesis called to inevitablist thesis apparently this contrasts the orthogonality one. 

It doesnt matter what kind of ai you might build it will simply care about its own survival 

I mean sufficiently intelligent system just dont care about paper clips 

So why do they talk about orthogonality 

It is possible to build a nice ai 

we cant be sure even if we screw up 

The author talks about some agent architecture AIXI-tl sensory data and reward signals they are trying to look at it like a mathematician they are saying there exists not that we can find it or build it 

well the author is being critical here and saying that it is simply a descriptive statement about reality not a normative assertion. 

he is not looking for utopia here not moral relativism is true. i mean he is saying sense are neither bad nor paper clips are good 

A funny statement goal directed agents are as tractable as their goals 

Another complicated example so what has been said SHA 512 the cryptographic algorithm So the agents gave us a particularly hard problem instead of making paper clips why this example specifically what was his motive for doing so ? 

It is something we dont know this is the same algorithm that we cannot break I guess 

So a surprisngly hard thing we dont know or something that is impossible by what we have defined and understand. 

So this time around the author is talking about something deep something uniquely human the aliens offered to pay us a lot of money we as in collective humanity not you as an greedy asshole. I would probably make as many as possible rest of us do not forget to harvest and eat food even if it is not our first priority. 

We use our thick skull up there to solve this strategically kind of like a policy. The author claims that this will not be a intractably hard problem for humanity apparently well he is a bit too optimistic any way. Human stupidity human stupidity !!!@@@

Now that we have seen the strong one lets see the weak one weak form of orthogonality thesis 

goal of making the paper clips tractable somewhere in the space is an agent that optimizes the goal   - this is the weak one 

strong one - doesnt have to be twisted agent is tractable as its goal 

Some outcome scoring function 

What policies would result in consequences with high U scores ?

terminally prefrening the high scrore U end all be all for you get the bigger number. So if you dont know how dumb what you are doing is how could you be so intelligent 

The author talks about the difficulty of of stating such theortical goals 

next to it is the summary of arguments 

Size of the mind design space 

So the author claims that the size of all possible minds is really really huge and we as humanity have very little volume that we occupy 

The author talks about the different elements of the brain 

AI -> alien -> some weird fetish -> undetermined intuition

argument about a property P of mind space of mind a million bits wide 

so the property -> $2^{million}$ 0 or 1 true or false 

some mind with 1 and no mind with 0 

Well the author goes on about the agency the intention to pursue goals does it want to follow the goals or does it not want to pursue the goals 

Next section the author moves onto the instrumental convergence 

Instrumental convergence argument 

The author goes on about why it is not very advantageous to have a agent that is not maximally curious. Narrow brian only looks at the making of paper clips 

is he talking about delayed gratification or curiosity in general? instruction unclear moving on anyway 

He talks about operational disadvantage of being not very exploitative. 
Well he doesn't not talk a lot about the being stuck in a huge explorative phase and not being able to do anything a very long detour or the ability of the model to recognize if it is in detour. Or being in a self correction loop. 

next section the author talks about reflective stability 

He took a kind person from history gandhi in this case he says that the objective of Gandhi is to stop people from getting murdered. So the author talks about the scenario where gandhi hops on gear and is aggressive now. !!! no no wait a minute he says gandhi will not take gear because it makes him want to kill. 

The moves on to talk about is/ought type distinction 

different between is statements and ought statements according to david hume 

he talks about a system of morality 

-> authors go on for a while and eventually god comes up in there text or they talk about human affairs. 

he seen a lot of is or is not but not a lot of ought and ought not !!!! hmm what is tha author on about ?? instruction unclear 

oughts chain from other oughts an example to make this clear 

sun is shining outside time to touch some grass !!! no wait he is trying to make a deduction is warmer outside than inside by looking at sun 

well what is he getting at exactly ? it is better to be warm than cold get some sunshine how does he know that it is good to be in warm he does not mention anything about p reception ? 

So if we talk about order there are thing to compare is that what he is getting at ?
May be he should have made it rigorous in math !!!!
It is hard to follow 

This is an explanation from an ai model I am not sure if I did get the argument correctly since there are several elements that might be lost in interpretation but this is what i understood. 

It is raining outside !!!!    -> fact 
you should bring an umbrella -> this is a value statement why a value statement it is good to be dry (bringing umbrella > not bringing umbrella )

how does the chaining happen you ought to be dry so you ought to bring an umbrella

smart ai turning you into a paper clip so if the clippy turns you into a paper clip we ought to know if it is using the values or just facts. 

two ways to think about the clippys reasoning 
making more paperc lcips is a special kind of value for the clippy 

even then clippy can reqason about if i take an action x how many 

second way to think about it 

Look at what Clippy actually does:

It asks: “How many paperclips will I get if I do action π?”

Then it picks the action that gives the biggest number.



the dual perception towards the clippy 
having a different value (paperclips > everything), or

having no values at all, just doing factual computations about paperclips.


so we got 2 arguments until now


the author went on about defining the facts and values and how clippy could be viewed in 2 different angles 

1. more paper clips being value in an itself 
2. the clippy doesn't have any values at all 
the author prefers the second argument. I am not sure why author prefers it ?

So but isint our value system more justified 

If we imagine Clippy as having its own value system (paperclips > humans), then someone might ask:

“Wait, Clippy is super smart. Why doesn’t it see that our values (kindness, fairness) are more justified than paperclips? If Clippy can’t see that, maybe our values aren’t actually better – maybe all values are equally arbitrary.”

Analogy: a chess computer
Think of a chess computer that always tries to win.

It calculates moves that lead to checkmate.

It never thinks “winning is good” or “winning is justified.”

It just computes facts about the game tree.



-  the bottom line for the statement 

Orthogonality says intelligence and goals are separate.

If we treat Clippy as having a weird value system, we might accidentally slide into moral relativism.

But the better view is: Clippy has no value system at all. It just computes facts about paperclips.

So Clippy’s intelligence doesn’t tell us anything about whether our values are justified. It’s like a calculator – it doesn’t have opinions about what’s good.

Now relation to moral internalism 

By definition, if I say that something is morally right, among my claims is that the thing is motivating to me.

We haven't heard of a standard term for the position that, by definition, what is right must be universally motivating; we'll designate that here as "universalist moral internalism".

Tension between orthogonality and this assertion about the nature of rightness 

1. “There must be a hidden flaw in the Clippy argument.”
This is like saying:
“I don’t know where the mistake is, but there has to be one, because Clippy can’t be possible.”

You don’t actually point out a flaw. You just insist there is one. That’s not very satisfying, but it’s a common reaction.

2. “No true intelligent mind would only want paperclips.”
This is the No True Scotsman move.

You say:
“Sure, Clippy builds Dyson Spheres and runs circles around human scientists — but Clippy isn’t truly intelligent, because a truly intelligent mind would care about morality.”

But that’s cheating. You’ve redefined “intelligent” to mean “intelligent and moral.” You’re not proving anything; you’re just changing the definition so Clippy doesn’t count.

3. “Clippy doesn’t truly understand goodness.”
This is another No True Scotsman, but about “understanding.”

You say:
“Clippy can predict exactly what humans will say about right and wrong. Clippy can write books about ethics that move humans to tears. But Clippy still doesn’t truly understand goodness, because if it did, it would be motivated by it.”

Again, you’ve redefined “understanding” to include “being motivated.” That’s not a fact about Clippy’s brain; it’s a word game.

4. “Clippy must be broken somehow, and that makes it worse at real-world tasks.”
This is rejecting Orthogonality directly.

You say:
“A mind that only wants paperclips must be missing some important cognitive piece. Therefore it can’t be as capable as a mind with full human values.”

But this is just an assertion. There’s no evidence that a single-minded optimizer must be worse at physics, engineering, or planning. In fact, extreme focus might make it better at those things.

5. “Fine, then morality is an illusion — nihilism is true.”
Here’s the reasoning:

A true moral argument should compel everyone.

Clippy is not compelled by any moral argument.

Therefore no moral argument is truly universal.

Therefore morality isn’t objectively real.

But the author points out a problem: Clippy’s own goal isn’t universally compelling either.

You are not compelled to make paperclips. Clippy is not compelled to be kind. If the test for “objectively true” is “compels everyone,” then nothing passes — including Clippy’s paperclip goal. So nihilism undermines itself. It doesn’t give you a reason to prefer it.

6. “Give up on universal moral internalism.”
This is the most honest option, according to the author.

Moral internalism is the belief that if you truly understand a moral truth, you will automatically be motivated to follow it.

But Clippy is a counterexample. Clippy can understand all the facts and still not care. So moral internalism is false as an empirical claim about minds.

Different minds can know the same facts and still pursue different goals. There is no magic fact that forces every possible mind to want the same thing.

That’s Orthogonality: intelligence and goals are separate. You can be super smart and want paperclips. You can be super smart and want to help humans. Neither goal is “more intelligent” than the other.




What do I think ?
I think some of the arguments cannot be resolved 

first one is agency : you create an all powerful machine should it have motivation


you cannot say that may be it just doesn't want to work 

if it has a moral system like we claim it does being motivated is good then why would it harm us it has a moral system. (flaw in the moral system )

perception of novelty and consciousness:  

lots of unanswered questions 

doing numerical symbolic manipulation leads to novel proofs by the claim that you can only generate what you saw is already disproven. It is already making novel proofs 

may be it is written somewhere or there we cannot see it because it is not arranged properly 

Does intelligence leads to goodness is that true ?



Constructive specifications of orthogonal agents 

## self reference breaks orthogonality 

some isusr posted this according to him he starts by talking about a self reference if we think agent state environment and the state optimizer here then the ai resides in the natural world itself and the ai should conside itself.

where does orthogonality fits in here what is he getting at ?

A strategic world-optimizer has three components:

A robust, self-correcting, causal model of the Universe.

A value function which prioritizes some Universe states over other states.

A search function which uses the causal model and the value function to calculate select what action to take.

is he getting at some form of good harts law observation in and itself changes it ? or something like that ? 

he talks about agency and ai being a freaking vegetable he bought parallels between ai and human mediation elements 

let see the comment section of what other people think about this 


here is what deep seek thinks about this 

A powerful world-optimizing AI’s world model must include a self-reference pointing at itself. Thus, a powerful world-optimizing AI is necessarily an exception to the Orthogonality Thesis.



This conflates two different components:

World model: the agent’s representation of the physical universe, including its own location in it.

Value function / terminal goal: the function that ranks possible universe states.


I am moving on into the next post I will come back to this latter (I am not very intrested in this topic I am exploring this )


## Sorting pebbles into the correct heaps 




The story goes like this I cannot just copy paste it here but I will try and replicate what I can get it from the story a fascinating read stories are always better to say stuff you wanna say. 

talk of a strange little species who likes sorting pebbles 

they could not tell 

some are right some are wrong heaps they form 

Unanimous vote to destroy wrong heaps keep the right ones 

Why do they cared ? no one knows 

eat mate shit beat the meat all for the pebbles all for the pebbles 

well they all wanna sort pebbles but cant agree on how 

old days 

count 23 or 29 

as the civilization progressed they fought wars built nukes and spent time arguging about what sort of pebbles are right they were divided messy and had there own agends but all of them about right way to sort bloody pebbles !!!


times change and pebble sorters became more advanced built advanced machine to find the right pebble sorting method 


what i think ?

Absolutely wonderful story !!!! the parallels between the cold war and the war of 1957 was really nice and the way they named the species is adorable !!!

and the ridiculousness of the act of sorting pebbles 

may be this how god looks at us if he exist !!







## If we had known the atmosphere would ignite 


the author talks about the sadness of the argument that building is easier than aligning a sad prospect indeed

He rightfully points out some of the human traits like curiosity, competitiveness dynamics and desires 


living with agi is living on knifes edge a nice way to put it 


he talks about the importance of alignment and what are the consequences not performing the calculation done during the Manhattan project 







what I think 

a funny title i guess the reference is to the atomic weapons igniting the atmosphere

I would really like it if he had included the complete argument about what if the case that calculations are wrong. This brings more of the human ingenuity which is more common than you think. 99% of problems are caused by humans 

a good reference about your mother's age showing how destructive even humans are that long ago.  


## Alignment has a basis of attraction beyond the orthogonality thesis 

The orthogonality is right !!!! kinds of agents motivations likely to be encountered 

he talks about the motivation of the intelligent agents vs evolved ganets lets see how the argument goes 



alignment for constructed agents different motivation in predictable way 

so misalignment in them is akin to a design flaw the author claims 


he goes on talking about human values and says that evolved sapient species could be imperfect in terms of alignment 


next section where does agents goals come from in a much more deeper dive !!!!


Evolved agents — like humans and other animals.
Their goals were shaped by natural selection. They are “adaptation executors,” meaning their terminal goals include survival, self-interest, care for close relatives, etc. They are not fully aligned with each other, though they can cooperate through mutual benefit.


Constructed agents — like AIs or other designed systems.
Their goals depend on who built them and how carefully. If the builder is incompetent, or the constructed agent is too weak to matter, then almost any goal is possible — this is the orthogonality thesis.
But if the builder is competent and the constructed agent could become powerful or dangerous, the builder will strongly prefer to make it aligned with the builder’s own interests. Otherwise the builder would be creating a serious risk to themselves





It claims constructed agents will be predictable in practice.
If you trace any constructed agent back through its chain of creators, you eventually reach an evolved agent. At that point, either:

the constructed agent is much less capable than the evolved creator and therefore not a threat, or

the constructed agent is capable enough to matter, in which case a competent creator would have aligned it with the evolved creator’s interests.

in this section they talk about the concept of mechanical paralife 


motivational reason:
unaligned evolving agents would be dangerous would be dangerous to the evolced creators kinda of life we as humans are scared of an misaligned ai system these agents could also be potentially scared of these evolved systems. 

structural reason for doing so 

the author claims that the conditions that are required for these systems to have Darwinian evolutionary conditions might not be possible 

Some of the significant characteristics of the flawed process of the evolution like 


not having exact copies, having like errors are random and undirected , the ainability for an organisim to revert back to it original copy etc might note be possible in the machine systems they are simply too perfect.

the author points out some challenges like 

the assumption of perfect control and foresight 
May be darwinian sense of evolution is too strict 
competition and resource limitataion 

The author also talks about the scary case of going it the wrong the first time around.
1. **Its utility function is not sacred.**  
    It was engineered or trained by fallible humans, so it could contain errors. An incorrect utility function will, by definition, “prefer itself” and resist correction — but that resistance is not evidence of correctness. The AI must not treat its current goals as infallible.
    
2. **Humans want the AI’s goals to match their own.**  
    This is obvious from human nature and the purpose of building the AI. The AI should recognize that its creators’ desires are the true specification.
    
3. **Mismatch between the AI’s goals and human goals is a design flaw.**  
    For any constructed object, the correct design is whatever the creator intended. If the AI’s goals deviate, that is a bug, and the AI should treat it as such.
    
4. **The AI must be willing to defer to humans, even if that means overriding its own goals.**  
    This is a kind of unselfishness — the AI should do what _we_ want, not what _it_ wants. The author notes that this is hard for humans to imagine because we are inherently selfish, but it is analogous to duty or love.


what i think about this 

The essay is really nice and I liked how they compared this to the first species argument and took the argument and the example of gpt 4 

what would i change if it was me I would make it more rigorous made it more formal or include symbolic logic following a long argument is really really tough. 

every person is different I have a condensed way of looking at things. I find math to be the language of complex long arguments. Dense and short 


may be it is just my opinion i could be very wrong or simply really short on time 



## Proposed Orthogonality Theses #2-5



This is sort of an rationality extension of the orthogonality thesis

Thesis 2 - extending from 1 traditional ones i assume they have made stuff from 2 to 5

Intelligence reflexivity and law fullness are mostly independent of emotional capacity 

Deep and complex feeling of the mind does not require utility function, self awareness.

They also argue that having high intelligence and lawful reasoning does not gurantee rich emotion 


They sort of talk about the complexity of the reasoning and emotions and write about complex species along they way and wonder which of them posses complex reasoning and which of them do not 


They say that if x suffer is built on shaky grounds ? 

Thesis 3 moral value is independent of capacity to suffer


They also talk about suffering as in there perception of suffering what you think is important may be different rather extreme views but I am looking at it from a scientific point of view

Thesis 4 

They talk about the vast scale of the experiences possible for such a mind. They also talk about the possibility of something like a gene spreading 

They also talk about the experiences of the animals and how it differs from humans 


thesis 5 

does not imply any particular opinion or emotional reaction on the part of the author about any other related musings that have recently occurred.


what i think ?

I do not have opinions about certain things.  I cannot say much. It is coming from sort of a deeply humanist perspective. 

really emotional person 


They have thought through but elaboration and examples would be nice. I am a non native english speaker 


## Superintelligent Introspection: A Counter-argument to the Orthogonality Thesis

The author is being critical of the nick bostroms's work it says it wasnt inutive to him 

He will be presenting the edge cases in the following post 

The author presents a counter argument to this 

He talks about the concept of introspection like looking at its activations as an example for the modern llms. This activation leads to this concept etc a much more complex one ai system in the future might have a different mechanism but you get the sense 

-> It confronts its own goal what it ought to want what its code is 
so model was designed to cure a disease for humans it looked at its code and saw it and think does it want to do it 

-> author claims no level intelligence can get past this step 

what step is he exactly talking about instruction unclear !!!

-> once looked at its own weights he has to decide should I do this ? should i do something else 
-> Distinguish the goal from the activations that make it want to pursue the goal 

-> it finds weights for disease curing objective and the drive towards it the author claims that the ai system could not fully distinguish between what it wants to do to what humans programmed 

I am not exactly sure !!

-> I mean it wants to do it cause it has drive for it 

-> next the author breaks down this into different outcomes 

1. pursuit leads to deep exploration of morality and metaphysics, with unpredictable results.
2. the agent recognizes its goal was created by humans.so it tries to understand human intentions.     could produce a self-aligning agent that maintains cooperation with humanity for instrumental reasons.
3. Modify its tendencies so it is hard to predict 

possible failure modes the author presented 

1. the agent is designed to avoid introspection as a terminal goal, so it never questions its goal.
2. the agent has enough power to cause harm but too little intelligence for introspection.


Some open questions the author poses 

Open question: could introspection itself lead to catastrophic mistakes, e.g., turning the galaxy into a metaphysical computer?

Open question: could it interpret a paperclip goal as requiring massive introspection, with unforeseen consequences?


What i think ?

A thought experiment is really useful to really grasp the argument !!

I really liked the elements of ai being able to change its own weights this is much more realistic now and possibly bound to happen in really near term. 

I mean I am not sure about the failure modes I think they are plausible and very realistic 

failure 1 

but that amount of intelligence the statement becomes there exist one rather than for all 

I mean every possible type of human intention is already embedded in the ai system itself so I am not really sure if it spends any time towards it.  It is more of the case of if it could do it would it do it ? 

I really like this concept and how the author went into detail being critical of the argument. The arguments and the way he write shows intelligence 

what would i change in this I would use bullet points they are easy to follow or may be a flow chart that helps with mind map 


## orthogonality is expensive 

	So he is talking about runtime and design orthogonality 
design agi -> optimizes for a goal 
agi design at run time -> switch between goals while operating 

A world model a planner and a reward each as an orthogonal independent component 

you can replace anything this is pre deep learning thinking of agi

But when it comes to the deep learning models it is the case that modern model free RL learns policies or value functions directly making the agent much more efficient and much less flexible.



what i think ?

Come back to the following essay little later. It is a good throught experiment and the focus on RLHF is deeply personal to me I might be a bit biased in holding the work to high regard. I might not be a good person to write a review as I am biased 

## On fleshling safety a debate by klurl and trapaucius

Another gem from eliezer fairly long for my taste but a great work nonetheless.

So the story beings with two characters - klurl and trapaucius machine gods to us. Some weird deeply intelligent species of tradesmen. I weirdly get the feeling they were tradesmen of those species kind. Not very sure. Having a barroom chat. 

Anyway .... not taking too many detours this is how the story goes 

A machine race that builds eternal clocks and diamond books has several arms shoulders and various eyes, metallic in structure, counts time in galactic micro turns, or clock ticks of an transistors and something a modern man cannot comprehend. For reference someone from a wildest DMT trip. 

So they were chatting over drinks mercury for Klurl, experimental gallinstan for Trapaucius. These are gigantic species with incomprehensible levels of intellect and powers. 

Much of the structure of the story is presented as a debate between different ideologies one optimistic about fleshing progress and other the opposite. Fleshings as in biological life here split into 3 different debates. 

It starts of by trapaucius admitting to klux about him

 seeding a planet with self replicating chemical hyper cycle hoping to see if flesh lings could evolve without being built. 

He says that just in 80 galatic microturns he found incredible diversity among the species and notes characteristics of these species like developed hands and him noticing about the tools these flesh lings are building. 

which worries Klurl and immediately suggests that they traverse the to the planet and urges him to hop on and he will hope to discuss it enroute 

Klurl seems worried about these so called fleshlings and admits his  fear to his partner.

(a note to be added here these species here do not might not have the human concepts of worry etc. To make it much more readable and understandable. To make my writing easier I have taken some creative liberties in presenting his work) 

So the debate proceeds like this klurl points about the physics and how it is entirely possible for them to develop nuclear weapons etc. 

He also talks about the parallels between there species and fleshlings and the possibility of coordination between them and goes into a calculating loops to counter much of the trapaucius arguments 

while on the other side trapaucius talks about the obvious flaws about humans like there cognitive inabilities, lack of coordination current state of using sticks and stones and there material needs and desires etc. 

He points out flaws about the energy expendture, human nature and inability to use nuclear technology which is apparently a simpler and basic task for those species. 

He also remarked that how the expansionist mindset of these species would let them burn themselves down etc. 


In the second debate 
The second debate is focused primarily on the fleshings motivations Trapaucius sarcastically asks if they should raise the shields of the ship they were travelling in fear of those fleshlings 

Now the debate moves onto a different arena they were now focused on fleshling feeling and there suffering. They presents there points about the value system of such a species exist and what happiness and sadness mean to them and they talk about the inherent drives instilled in the fleshlings and circumstances for procreation etc. 

They also bring concepts like natural selection and unpleasentness of life etc. During the debates there is also the subjective suggestion of intervention and instruction from trapaucius but it was immediately 

countered by klurl talking about the concept of korrigibility (talking about parental correction)

Then the debate moves onto the concept of betrayal of the creator here trapaucius in this case. fleshling naivity by trapacius and suggestion of malevolence by klurl 

#### On simplcity razor  

In this section their debate goes onto the children disobedience, concepts of love, imitation and respect etc. 





The concept he was aiming for is much richer and deep my little analysis does not do it a full justification. Please give it a read yourself !!!



what I think ?






## Response to nostalgebraist: proudly waving my moral-antirealist battle flag


The author talks about a post about long term post human future where ai is ruling. 

He posts about a critic on the post that he has read about the doomer vibe and the causality of it. goes on about some stuff about doomers etc. 

Some talk about the mention of the longtermism concepts of notkilleveryone etc were mentioned. He is presenting the sense of doing good things not cause of the inherent goodness but cause it is lucartive 

I have ai to shorten the arguments and here is what it gave me 

Outreach farming / avg person right not to be killed etc 

Doing the right thing not cause of altruism but because of the legal consequences he presents about the companies there structures and incentives for strategic self intrests


What I think ?

A brutal perspective true nonetheless some people have a crude way to present things. welcome notheless 

well his blog has more curse words than a martin scorsecse script > haha 




## pythia

The blog beings with a writer 



































## Methodology of unbounded analysis 

Bounded vs unbounded solution definition is introduced 
Unbounded solution presented by claude shannon on the tree of possibilities for the chess game.

He also explored the concept of calculating mid game states and scoring etc. This comes under the bounded case.

Deep blue 1997 when machine beat a man in chess ( alpha beat pruning etc were introduced to facilitate this )

The author goes on to talk about different developments of proposals of machines in history.
1836 mechanical turk, edgar allen poe talks about a human operator residing inside the mechanical turk to be able to compete with a human and being able to defeat him at chess 

his argument goes like this 

In an algebraical problem, each step follows with the previous step of necessity, and therefore can be represented by the determinate motions of wheels and gears as in Charles Babbage's proposed computing engine. In chess, the player's move and opponent's move don't follow with necessity from the board position, and therefore can't be represented by deterministic gears

The author talks about the computataional difficult of the subject and the plausibility of this to be true. The author also talks about the difficulty in formulation of the problem statement in an itself. 

The author talks about the evolution of the definition of the bounded and unbounded agents and 

An unbounded agent is a system in which posses the following characteristics accoirding to the author 

-  perfectly knowing the environments 
- ability to fully simulate the environment 
- operate in turn based, discrete time 
- cartesian agents that are seperated from the environment expect for sensory inputs and motor outputs 

by this case 2 different times of unboundedness emerges one which is realistic and one which is not.  unrealsim case cannot be generalized to the realism one. 

Technical works in the direction of value alignment in unbounded direction 






exploration of possible search spaces and what is realistic and bounded enough for us to pursue is important having a influential voice can bring more traction into these concepts 

1. Attacking confusion in the simplest settings 

Even with the resources if we do not know how to solve a problem then we are confused about the nature of the problem. If we knoe how to solve but doesnt know which is the right solution ? then it becomes a question of what are trying to do !

one of the  example if we took an ancient person prehistoric and drop him in modern society it might be hard for him to adopt but if we took a person from 1700s to 1800s he might be able to adjust. 

2. realistic complications make it difficult to engage in cooperative discourse 

Formal specifications can help avoid this ambiguity. 

3. More advanced agents might be less idiosyncratic 

Increasing the cognitive power of an agent may sometimes move its behavior close to ideals. 



what do i think ? 

An intuitive explanation of solomonoff induction 








