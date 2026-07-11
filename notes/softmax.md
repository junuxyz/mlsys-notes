# Softmax: From Naive to Blocked Softmax

What is Softmax? The softmax function is a function that converts a vector of length $K$ into a probability distribution of length $K$.

For a given an input vector

```math
\mathbf{z} = [z_1, z_2, \dots, z_K]
```

the softmax value for the $i$-th element is defined as

```math
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
```

where
- $\sigma$ = softmax function
- $\mathbf{z}$ = input vector; size of (1, K)
- $K$ = length of the vector row.
- $z_i$ = score of the $i$-th class
- $\sigma(\mathbf{z})_i$ = probability of the $i$-th element softmax
- $e^{z_i}$ = exponent of $i$-th element $z_i$
- $d_K = \sum_{j=1}^{K}$ = denominator that adds the entire row. This makes the sum of all $\sigma(z)_i$ to 1.


Computationally this happens in two passes:
1. first pass: compute the denominator $d_K = \sum_j^K e^{z_j}$ (need to sum all elements)
2. second pass: divide each $e^{z_i}$ by the denominator $d_K$

> [!NOTE]
> **Why do we use exponent $e$?**
>
> We use $e \approx 2.718$, which is known as [_Euler's number_](https://en.wikipedia.org/wiki/E_(mathematical_constant)).
>
> The main reason we use $e$ is
>
> 1. differentiating $e^x$ result in $e^x$ so it's convenient so it's mathematically convenient.
> 2. to make any arbitrary elements in $z$ to be positive value.

### Safe Softmax

While softmax can convert vector into probability, the normal softmax formula can be numerically unstable on computers.

To be more concise, some $z_i$s in $z$ can be large (e.g. 1,000) which becomes $e^{1000}$. In computers, this is about

```math
e^{1000} \approx 10^{434.294} \approx 1.97 \times 10^{434}
```

which is too large ($\approx \infty$) to be represented in floating point operation. This it is important to cap the upper bound in order to safely calculate softmax.

Safe Softmax is a simple idea to subtract the maximum logit before applying exponential.

```math
\sigma(\mathbf{z})_i
=
\frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}
```

If we compare this with vanilla softmax above, we just subtracted the element that has the maximum value in vector $z$. A minimal example would be to convert

```
[e^0, e^1, e^1000, e^5] -> [e^-1000, e^-999, e^0, e^-995]
```

This does not change the final softmax result because subtracting the same constant from every logit does not change their relative differences. It only makes the computation numerically safe.

However unlike naive softmax, this requires an additional pass:
1. **first pass: find the maximum value $m$**
2. second pass: compute the denominator $\sum_j^K e^{z_j - m}$ (need to sum all elements)
3. third pass: finally divide each $e^{z_i - m}$ by the denominator


**Proof**

Using the rule
```math
e^{a-b} = e^a e^{-b}
```

we get

```math
\sigma(\mathbf{z})_i
=
\frac{e^{z_i} e^{-m}}{\sum_{j=1}^{K} e^{z_j} e^{-m}}
```

Since $e^{-m}$ does not depend on $j$, we can factor it out from the denominator:

```math
\sigma(\mathbf{z})_i
=
\frac{e^{z_i} e^{-m}}{e^{-m} \sum_{j=1}^{K} e^{z_j}}
```

Now $e^{-m}$ appears in both the numerator and denominator, so it cancels out:

```math
\sigma(\mathbf{z})_i
=
\frac{e^{z_i} \cancel{e^{-m}}}{\cancel{e^{-m}} \sum_{j=1}^{K} e^{z_j}}
```
```math
\sigma(\mathbf{z})_i
=
\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
```

Therefore, we can guarantee subtracting $\max(\mathbf{z})$ does not change the softmax result.

### Online Softmax

Introduced in NVIDIA's paper [*Online Normalizer Calculation for Softmax*](https://arxiv.org/pdf/1805.02867), online softmax is a way that computes the same softmax result, but in a more _efficient_ way.

Safe softmax required us to read and go through the same vector three times. Can we read fewer than this?

Online softmax improves this by computing the maximum value and the denominator target together (step 1 and 2 above) **in one pass**, which leads to two passes in total:
1. first pass: compute the denominator $d_i = \sum_j^K e^{z_j - m_i}$ and iteratively rescale the past sum based on the new maximum value (explained below)
2. second pass: divide each $e^{z_i - m_K}$ by the denominator

In order to do this, we maintain two running values during the first pass:

```math
m_i = \max(m_{i-1}, z_i)
```
```math
d_i = d_{i-1} e^{m_{i-1} - m_i} + e^{z_i - m_i}
```

where
- $m_i$ is the _running_ maximum up to position $i$
- $d_i$ is the _running_ denominator up to position $i$

What if a new maximum appears at $i$-th index?

The previous denominator $d_{i-1}$ is rescaled by $e^{m_{i-1} - m_i}$

**Rescaling**

We can represent the previous running denominator as

```math
d_{i-1}
=
\sum_{j=1}^{i-1} e^{z_j - m_{i-1}}
```
and by using the exponential rule,
```math
e^{z_j - m_{i-1}} = e^{z_j} e^{-m_{i-1}}
```
we get

```math
d_{i-1}
=
e^{-m_{i-1}}
\sum_{j=1}^{i-1} e^{z_j}
```
The goal we wanted to do in safe softmax was for all denominators to be

```math
d_{i}
=
\sum_{j=1}^{i-1} e^{z_j} e^{-m_{i}}
```
so we need to rescale the old denominator $d_{i-1}$ to match the new maximum value $m_i$ and then add the current element's exponent.

This can be written as
```math
d_i
=
d_{i-1} e^{m_{i-1} - m_i}
+
e^{z_i - m_i}
```


After scanning the whole vector, we get the global maximum value and global sum (by rescaling) in one pass!

```math
m_K = \max(\mathbf{z})
```
and
```math
d_K = \sum_{j=1}^{K} e^{z_j - m_K}
```
The second pass works the same as third pass of safe softmax (divide each $e^{z_i - m_K}$ by the denominator):
```math
\sigma(\mathbf{z})_i = \frac{e^{z_i - m_K}}{d_K}
```


### Blocked Softmax

**Blocked softmax** is the block-level version of online softmax. Instead of updating the running maximum and denominator one element at a time, we divide the vector into blocks and compute a local maximum and local denominator for each block.

Suppose the input vector $\mathbf{z}$ is divided into $B$ blocks:

```math
\mathbf{z} = [\text{block}_1, \text{block}_2, \dots, \text{block}_B]
```

For each block $b$, we compute the local maximum:

```math
m_b = \max_{j \in \text{block}_b} z_j
```

> [!NOTE]
> **Do we need to read the local blocks once more to find the local maximum, as in safe softmax?**
>
> Yes but not in the same "expensive" way.
> In safe softmax, finding the maximum requires a separate pass over the entire row from global memory. Then we need another pass to compute the denominator using that maximum.
> In blocked softmax, each block is loaded from HBM/global memory once, and then the local maximum and local denominator are computed from the loaded values while they are still on-chip.
>
> In pseudocode:
> ```
> load block once from HBM/global memory
> 	compute local max from loaded values
> 	compute local denominator from those same loaded values
> ```
>
> So logically, we still need the local maximum before computing the local denominator but memory-wise, we do not need to reread the same block from HBM just to find the local max. The block values can be reused from faster on-chip storage such as registers or shared memory.


The local denominator normalized by that local maximum:

```math
d_b = \sum_{j = 0}^{\text{block size} - 1} e^{z_j - m_b}
```
So each block produces a pair:

```math
(m_b, d_b)
```
where
- $m_b$ = the largest value inside the block
- $d_b$ = the local exponential sum after subtracting that local maximum

Now we need to merge these block-level results into one global result. Let $(M_b, D_b)$ be the running global maximum and denominator after processing blocks $1$ through $b$.

The running maximum is updated as:

```math
M_b = \max(M_{b-1}, m_b)
```

The running denominator must be rescaled (same logic as online softmax) whenever the maximum changes:
```math
D_{b-1} e^{M_{b-1} - M_b}
+
d_b e^{m_b - M_b}
```

After all blocks are merged, we get:

```math
M_B = \max(\mathbf{z})
```

and

```math
D_B = \sum_{j=1}^{K} e^{z_j - M_B}
```

Then the final softmax value is computed as:

```math
\frac{e^{z_i - M_B}}{D_B}
```


The full code from naive softmax to blocked softmax can be found [here](https://github.com/junuxyz/mlsys-notes/blob/main/labs/flash_attn/softmax.py)(~100 lines of easy python code). It will be very easy to read after you've read this section.
