---
layout: post
title: "Forward-Backward Algorithm"
tags: [HMM, forward-backward]
comments: true
share: true
mathjax: true
---

Forward-backward algorithm (FB) is a particular case of a dynamic programming. This algorithm is well known in the context of Hidden Markov Models (HMM) where it is used for training and inference, Kalman Smoothers and Connectionist Temporal Classification [(CTC)](http://www.cs.toronto.edu/~graves/icml_2006.pdf). 
It consists of two passes: the first pass goes forward in time and second pass goes backward, hence the name.

FB algorithm heavily relies on Markov property and output independence assumptions from HMM. Actually a lot of terms here are inherited from HMM, so if you're not familliar then this [HMM tutorial](http://cs229.stanford.edu/section/cs229-hmm.pdf) might come in handy.

Suppose that $$z=\{z_1,z_2,..,z_n\}$$ is hidden variable, $x=\{z_1,z_2,..,z_n\}$ is observable variable and $k=1..n$ denotes timestep, then

* $$p(x_k\vert z_k)$$ is an Emission probability;
* $p(z_k\vert z_k-1)$ is a Transition probability;
* $p(z_1)$ is an Initial probability.

To simplify the notation let's further assume that hidden state variable $z_i$ defined on an interger set $\{1,..,m\}$

The goal of FB algorithm is to compute $p(z_k\vert x)$.

The forward pass computes: $p(z_k,x_{1:k}) \forall k=1,..,n$

The backward pass computes: $p(x_{k+1:n} \vert  z_k) \forall k=1,..,n$

For now let's assume that we've already computed forward and backward parts, in other words we know $p(z_k,x_{1:k})$ and $p(x_{k+1:n} \vert  z_k)$. How does it help us to get desired $p(z_k\vert x)$?

$$
\begin{align}
p(z_k\vert x) \propto p(z_k,x) &= p(z_k, x_{1:k}, x_{k+1:n}) \\
&= p(x_{k+1:n}\vert x_{1:k}, z_k) p(z_k, x_{1:k}) \\
&= p(x_{k+1:n}\vert z_k) p(z_k, x_{1:k})
\end{align}
$$

The last step applies conditional independence property ($x_{k+1:n}$ is conditionally independent of $x_{1:k}$). After simplification we have: 

$$p(z_k\vert x) \propto p(x_{k+1:n}\vert z_k) p(z_k, x_{1:k}) $$

And we already know that $p(z_k, x_{1:k}), p(x_{k+1:n}\vert z_k)$ can be found with forward and backward algorithms. The proportionality is instead of equality is due to omitting normalizing constant $p(x)$ which we can compute by marginalizing $p(x,z_k)$ over all finite set of $z_k$:
$$p(x) = \sum_{z_{k}=1}^{m} p(x,z_k)$$


## Forward algorithm

$$
\begin{align}
p(z_k, x_{1:k}) &= \sum_{z_{k-1}=1}^{m} p(z_k,z_{k-1},x_{1:k}) \\
&= \sum_{z_{k-1}=1}^{m} p(x_k\vert z_k,z_{k-1},x_{1:k-1})
                     p(z_k\vert z_{k-1},x_{1:k-1})
                     p(z_{k-1},x_{1:k-1})
\end{align}
$$

After applying Markov properties we get:
$$p(z_k, x_{1:k})
= \sum_{z_{k-1}=1}^{m} p(x_k\vert z_k) 
                     p(z_k\vert z_{k-1}) 
                     p(z_{k-1},x_{1:k-1}) $$

* $p(x_k\vert z_k)$ is given as an emission probability 
* $p(z_k\vert z_{k-1})$ is given as a transition probability 
* $p(z_{k-1},x_{1:k-1})$ is unknown, but you can notice the recurrence here if you'll look at what we are trying to compute: $p(z_k, x_{1:k})$

To find $p(z_k, x_{1:k})$ we need to compute $p(z_{k-1},x_{1:k-1})$, for $p(z_{k-1},x_{1:k-1})$ you have to compute $p(z_{k-2},x_{1:k-2})$ and so on until $p(z_{1},x_{1})$. And $p(z_{1},x_{1})$ is just an emission probability that is known. Now we know all the parts to achieve our goal finding $p(z_k, x_{1:k})$. 

What about time complexity? We have $n$ timesteps ($k=1..n$) and inside each timestep we iterate over $m$ values of $z_k$ and $m$ values of $z_{k-1}$, this gives us $O(nm^2)$ time complexity.
                     

## Backward algorithm

$$
\begin{align}
p(x_{k+1:n}\vert z_k) &= \sum_{z_{k+1}=1}^{m} p(x_{k+1:n}, z_{k+1} \vert  z_k) \\
&= \sum_{z_{k+1}=1}^{m} p(x_{k+2:n} \vert  x_{k+1}, z_{k+1}, z_k)  p(x_{k+1}\vert z_{k+1}, z_k) p(z_{k+1}\vert z_k) 
\end{align}
$$

And again after simplification with Markov assumptions we get:

$$ p(x_{k+1:n}\vert z_k) = \sum_{z_{k+1}=1}^{m} p(x_{k+2:n} \vert  z_{k+1}) p(x_{k+1}\vert z_{k+1}) p(z_{k+1}\vert z_k) $$

* $p(x_{k+2:n} \vert  z_{k+1})$ can be computed recurrently
* $p(x_{k+1}\vert z_{k+1})$ is given as an emission probability 
* $p(z_{k+1}\vert z_{k})$ is given as a transition probability 

We compute $p(x_{k+2:n} \vert  z_{k+1})$ recurrently until we reach $p(x_n\vert z_{n-1})$. We can do the same routine as previously:

$$ 
\begin{align}
p(x_n\vert z_{n-1}) &= \sum_{z_n=1}^{m} p(x_n,z_n\vert z_{n-1}) \\
&= \sum_{z_n=1}^{m} p(x_n\vert z_n,z_{n-1}) p(z_n\vert z_{n-1}) \\
&= \sum_{z_n=1}^{m} p(x_n\vert z_n) p(z_n\vert z_{n-1}) 
\end{align}
$$

This formula is a special case of previous more general formula where the term $p(x_{k+2:n} \vert  z_{k+1})$ has become equal to 1, and if $k=n-1$ theh we'll get $p(x_{n+1} \vert  z_n) = 1$.

Summarizing all that in the final formula:

$$
p(x_{k+1:n}\vert z_k) = \sum_{z_{k+1}=1}^{m} p(x_{k+2:n} \vert  z_{k+1}) p(x_{k+1}\vert z_{k+1}) p(z_{k+1}\vert z_k) \\
p(x_{n+1} \vert  z_n) = 1
$$

This is quite similar to forward part and time complexity is also $O(nm^2)$ because we have $k=1..n$ timesteps, $m$ values of $z_k$ and $z_{k+1}$ that we iterate through.

## Naive approach

To better understand the value of FB algorithm let's just compare it with the naive approach. With conditional probability formula we know that: 

$$p(z_k \vert x) = \frac{p(x,z_k)}{p(x)} \propto p(x,z_k) $$

We can try directly compute $p(x,z_k)$ and for that we have to take into account all possible sequences of $z$ where $z_k$ might have been occured or in other words we should marginalize out all except $z_k$: $\hat{z} = z_1..z_{k-1},z_{k+1}..z_n$ variables:

$$ 
\begin{align}
p(x,z_k) &= \sum_{\hat{z}} p(x,\hat{z}) \\
&= \sum_{z_1} \sum_{z_2} ... \sum_{z_{k-1}} \sum_{z_{k+1}} ... \sum_{z_{n}} p(x,\hat{z}) 
\end{align}
$$

The number of sums is equal to number of possible permutations with repetitions of hidden state sequence $\hat{z}$ which is equal to $m^{n-1}$. This is an exponential time complexity, for $T = 101$ steps and $m = 10$ hidden states it will have $10^{100}$ terms to compute. While Forward-Backward algorithm will have $m^2 \times n = 100\times101 \approx 10^4$, that is $10^{96}$ times faster!

<!--
References:
http://www.andrew.cmu.edu/user/scheines/tutor/d-sep.html
-->