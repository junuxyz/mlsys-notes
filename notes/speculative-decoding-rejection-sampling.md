# rejection sampling explained

It took me a while to understand rejection sampling. It almost made sense, but I couldn't explain why it reproduces the target model's distribution exactly. I've seen many analogies for how it works, but not a clear mathematical explanation of why it's guaranteed to match the target model's distribution (edit: I later found [this blog post](https://jwlabs.vercel.app/post/speculative-decoding-first-principles), which explains rejection sampling very well!). So in this note, I'll briefly go through the math intuitively.

## how sampling works

Before we talk about sampling in speculative decoding, let's first understand what sampling is and how an autoregressive model does it.

TL;DR: model forward pass (decoder layers) -> next-token logits -> softmax -> sample the next token

> [!NOTE]
> We'll go over this section briefly, leaving out top-k/top-p sampling and temperature scaling, since sampling methods aren't our main focus. If you're interested in sampling itself, I recommend [this resource](https://github.com/navaneethkrishnansuresh/Inference-Engineering/blob/main/basic_implementation/02_sampling/02_sampling.ipynb), which I found helpful.

### next token logits

After passing through the decoder layers, each of which usually contains an attention module and an FFN module, we get a final hidden state of size `[H]`. After a final layer norm, we pass it through the LM head, whose weight matrix has shape `[H, vocab]`. The LM head projects the hidden state into a vector the size of the vocabulary (e.g. 120k). These scores are called _[logits](https://docs.lm-kit.com/lm-kit-net/guides/glossary/logits.html)_, with one score for each token in the vocabulary. A higher score means the model considers that token more likely given the previous context. We haven't applied softmax yet, so the values are not bounded.

### softmax

We apply softmax to convert the logits into probabilities. The probabilities add up to 1, and each one lies between 0 and 1.

### next token sample

After softmax, we have a categorical distribution over the vocabulary - a vector of 120k numbers summing to 1. Sampling means drawing one index from that vector, weighted by its probability, using a source of randomness.

## greedy decoding vs sampling

Greedy decoding always chooses the token with the highest probability (`argmax`) at each position, so it's deterministic once the logits are formed. Sampling, on the other hand, can choose different tokens in the same situation. You can think of it as a "weighted random choice" biased toward tokens with higher probabilities.

## greedy decoding in speculative decoding

Let's first look at how greedy decoding works in speculative decoding.

We draft tokens (e.g. 8), then the target verifies them using the draft tokens as input and samples the next token. Verification is simple because we just need to compare the draft and target argmax tokens and see whether they match. The target accepts tokens until the draft and target choices diverge (because argmax makes the choice deterministic), then selects its own token at that position.

For example, if the draft and target `argmax` tokens match at the first four positions but diverge at the fifth, the target accepts the first four tokens and chooses a replacement for the fifth, giving us five tokens in total. Since we're choosing the highest-probability token at each position, we don't need residual sampling in greedy decoding.

## sampling in speculative decoding

As explained above, sampling works differently. We decide whether to accept a token proposed by the draft model using rejection sampling.

In the context of speculative decoding, $q(x)$ is the draft model's probability for token $x$, and $p(x)$ is the target model's. Both distributions come from applying softmax to the logits.

**[Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)** lets us draw samples from a target distribution $p(x)$ using a cheaper distribution $q(x)$ to propose tokens. We use $p(x)$ to decide whether to accept each proposal and correct for rejected proposals by resampling from what's left over. This method guarantees the final output is distributed exactly as if it had been sampled from the target model alone.

Concretely, after drafting token $x$ from $q(x)$, we accept it with probability

$$
a(x) = \min\left(1, \frac{p(x)}{q(x)}\right).
$$

If rejected, we resample from the residual distribution

$$
p'(x) = \frac{\max(0,\,p(x)-q(x))}{\sum_i \max(0,\,p(i)-q(i))}.
$$

The next section shows why these rules reproduce $p(x)$ for every token in the vocabulary.

### mathematical intuition of how rejection sampling works

The goal here is to show that the final output token, produced through this whole accept, reject, resample process, is distributed exactly as $p(x)$ even though a cheaper draft $q(x)$ is doing the proposing.

The best way I found to describe this is the equation below (believe me, this is the best equation I've seen while trying to understand speculative sampling):

For any token $x$ in the vocabulary:

$$
P(\text{output}=x)
=P(x\text{ accepted})+P(x\text{ delivered via rejection and resampling}).
$$

Here, the _accept term_ is $P(x\text{ accepted})$, and the _residual term_ is $P(x\text{ delivered via rejection and resampling})$.

For speculative sampling to match $p(x)$, these two terms must add up to $p(x)$. These are the **only two paths** by which $x$ can become the output: either directly or after rejection and resampling.

Now let's check the two cases.

### case 1. undersampled

$p(x) > q(x)$

Say $q(x)=0.2$ and $p(x)=0.5$, which means the draft proposes $x$ less often than the target would. If the draft happens to draw $x$ anyway, it's accepted 100% of the time. Why is that safe? The accept term cannot contribute more than $q(x)$, and $q(x)$ is smaller than $p(x)$. In other words, accepting every proposal of $x$ contributes only $0.2$, so it cannot overshoot the target's $0.5$. The remaining $0.3$ of total output probability comes from rejection and resampling.

> [!NOTE]
> Q. Is accepting all of $q(x)=0.2$ and supplying the remaining $0.3$ through residual sampling the only valid split? What if we accept fewer proposals of $x$?
> 
> A. No. We could accept only $0.15$ of the $0.2$, as long as the missing $0.35=0.5-0.15$ is supplied through a correspondingly adjusted residual distribution. But accepting less than the maximum creates _more_ rejection events without improving the final distribution.
> 
> In short, multiple splits can reproduce the target distribution, but assigning as much probability as possible to acceptance avoids unnecessary rejections. That's why the acceptance rule uses $\min(q(x),p(x))$.

### case 2. oversampled

$p(y) < q(y)$

This time, say $q(y)=0.5$ and $p(y)=0.2$, which means the draft proposes token $y$ more often than the target would. We cannot accept every proposal of $y$, since that would give it too much output probability. Instead, we accept it with probability $p(y)/q(y)=0.2/0.5=0.4$.

What does this $0.4$ mean? When $y$ is proposed, it has a $0.4$ chance of being accepted. Since $y$ is proposed with probability $0.5$, its total accepted probability is $0.5 \times 0.4=0.2=p(y)$! So $y$ can still be accepted even when $q(y)>p(y)$.

Let's say we land in the other 60% and reject $y$. We then resample, but not directly from the target distribution. We use the normalized residual distribution $p'(i) \propto \max(p(i)-q(i),0)$, where $i$ ranges over all vocabulary tokens.

Why $p(i)-q(i)$?

When $q(i)>p(i)$, we accept proposed token $i$ with probability $a(i)=p(i)/q(i)$, so $P(i\text{ accepted})=p(i)$. Resampling $i$ would overshoot its target probability, so its residual weight is zero.

Remember $x$ from Case 1, still $0.3$ short of its target. A rejected proposal, including a rejected $y$, can produce $x$ through residual sampling. Across all rejections, this supplies the missing $0.3$ of total output probability. The residual distribution therefore assigns weight only to tokens whose target probability exceeds the probability already supplied by accepted proposals.
