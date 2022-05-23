# Restricted Boltzmann Machines and Contrastive Divergence

A **restricted Boltzmann machine** is an undirected graphical model with two layers: one layer of visible binary variables $\bm v$ and one layer of latent binary variables $\bm h$. Intralayer connections are forbidden, forming a bipartite graph structure.

![RBM Drawing](img/rbm_drawing.png)

The model defines the joint probability distribution:
$$
    P(\bm v, \bm h) = \frac{1}{Z} \exp( -E(\bm v, \bm h))
$$
where $E(\bm v, \bm h) = -\bm b^\top \bm v - \bm c^\top \bm h - \bm v^\top \bm W \bm h$ is the **energy** function. The parameters $\bm b$ and $\bm c$ are bias vectors for the visible and hidden variables, respectively. $\bm W$ is the weight matrix representing the connections between $\bm v$ and $\bm h$. $Z$ is the **partition function**, the normalizing constant (with respect to $\bm v$ and $\bm h$) that ensures $P$ is a probability distribution by making $P$ sum to 1 over all $\bm v$ and $\bm h$.
$$
    Z = \sum_{\bm v} \sum_{\bm h} \exp(-E(\bm v, \bm h))
$$

Note that if $\bm v$ or $\bm h$ are high-dimensional (as in most cases), $Z$ becomes intractable.

## Gibbs Sampling

Because of the intractability of $Z$, exact samples can not be drawn from $P(\bm v, \bm h)$. Instead, $P(\bm v, \bm h)$ is approximated using **Gibbs sampling**.

In the most general form, Gibbs sampling is used to sample an $n$-dimensional sample $\bm x = \langle x_1, \dots, x_n \rangle$ from a joint probability distribution $P(\bm x)$. It is often used when the joint probability distirbution is difficult or impossible to sample from, but the conditional probability distributions are easy. The algorithm is as follows:

1. Initialize the initial sample $\bm x^{(0)}$ to some value (usually randomly or by some heuristic/algorithm).
2. Update the sample $\bm x^{(t)} \to \bm x^{(t + 1)}$ by sampling each $x_i^{(t+1)}$ individually from the conditional distribution $P(x_i^{(t+1)} \mid \mathbb X)$, where $\mathbb X = \{ x_1^{(t)}, \dots, x_n^{(t)} \} \setminus \{ x_i^{(t)} \}$ is the set of individual samples from the previous iteration, excluding the individual sample corresponding to $x_i^{(t+1)}$.
3. Perform $k$ iterations of the above step.

This process describes a Markov chain, where the state of the Markov chain is given by $P(\bm x^{(t)})$ and the transition transformation $T$ is described in Step 2. When the Markov chain reaches a stationary distribution, $x^{(t)}$ becomes an exact sample from $P(\bm x)$. The number of iterations $k$ required for the Markov chain to converge to a stationary distribution is called the **mixing time**. Running the Markov chain until convergence is known as **burning in** the chain. Because $P(\bm x)$ is usually intractable, It is currently unknown how to determine the mixing time of an RBM's Markov chain or to know when the Markov chain is mixed. So, most implementations halt after some constant number of iterations, say $k$. Although the sample is not exact, it is usually a good enough approximation.

### Sampling Restricted Boltzmann Machines with Gibbs Sampling

A sample of $\bm v$ and $\bm h$ can be drawn from an RBM using Gibbs sampling. Because intralayer connections are prohibted in an RBM, an observed variable $v_i$ is conditioned only on $\bm h$. So, all $v_i \in \bm v$ are conditionally independent given $\bm h$. Therefore, rather than sampling each $v_i$ individually, the entire vector $\bm v$ can be sampled in parallel from $P(\bm v \mid \bm h)$ in Step 2 of the Gibbs sampling algorithm. Similarly, the entire vector $\bm h$ can be sampled in parallel from $P(\bm h \mid \bm v)$. Simultaneous sampling of conditionally independent variable while Gibbs sampling is called **block Gibbs sampling**.

The conditional probabilities for both are derived as follows:
$$
\begin{align*}
    P(\bm h = \bm 1 \mid \bm v)
    &= \frac{P(\bm v, \bm h = \bm 1)}{P(\bm v)} \\
    &= \frac{P(\bm v, \bm h = \bm 1)}{P(\bm v, \bm h = \bm 1) + P(\bm v, \bm h = \bm 0)} \\
    &= \frac{\exp(\bm b^\top \bm v + \bm c + \bm v^\top \bm W)}{\exp(\bm b^\top \bm v + \bm c + \bm v^\top \bm W) + \exp(\bm b^\top \bm v)} \\
    &= \frac{\exp( \bm c + \bm v^\top \bm W)}{\exp(\bm c + \bm v^\top \bm W) + 1} \\
    &= \sigma \left( \bm c + \bm v^\top \bm W \right) \\
    P(\bm h = \bm 0 \mid \bm v) &= 1 - \sigma\left(\bm c + \bm v^\top \bm W\right) \\
    &= \sigma\left(-\bm c - \bm v^\top \bm W\right)
\end{align*}
$$

and symmetrically,
$$
\begin{align*}
    P(\bm v = \bm 1 \mid \bm h) &= \sigma \left( \bm b + \bm W \bm h\right) \\
    P(\bm v = \bm 0 \mid \bm h) &= \sigma \left( -\bm b - \bm W \bm h\right)
\end{align*}
$$

Note that when computing $P(\bm v \mid \bm h)$, it is important that $\bm h$ represents the binary states of the hidden units and not their real-valued probabilities. This acts as a regularizer, as it prevents the hidden units from communicating real-values to the visible units during reconstruction [(Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf).

## Training Restricted Boltzmann Machines with Likelihood Maximization

RBMs can be trained using likelihood maximization. Let $\bm x$ be the vector of variables in the model (including both observable variables and latent variables $\bm v$ and $\bm h$) and $\bm \theta$ the vector of parameters. Likelihood maximization is achieved using gradient ascent of the log-probability of the variables $\bm x$ with respect to the parameters $\bm \theta$. This gradient ascent requires computing the following gradient:
$$
    \nabla_{\bm \theta} \log P(\bm x; \bm \theta) = \nabla_{\bm \theta} \log \tilde P(\bm x; \bm \theta) - \nabla_{\bm \theta} \log Z
$$
where $\tilde P(\bm x) = \exp(-E(\bm v, \bm h))$ is the unnormalized probability distribution. The positive and negative terms of this decomposition are called the positive and negative phases of learning, respectively.

The partial derivatives for the positive phase are given by:
$$
\begin{align*}
    \frac{\partial}{\partial \bm W}\left( -E(\bm v, \bm h)\right) &= \bm v \bm h^\top \\
    \frac{\partial}{\partial \bm b}(-E(\bm v, \bm h)) &= \bm v \\
    \frac{\partial}{\partial \bm h}(-E(\bm v, \bm h)) &= \bm h
\end{align*}
$$

The partial derivatives for the negative phase are more difficult because of the intractability of $Z$. Instead, we work around $Z$ using the following identity:

$$
\begin{align*}
    \nabla_{\bm \theta} \log Z
    &= \frac{\nabla_{\bm \theta} Z}{Z} \\
    &= \frac{1}{Z} \nabla_{\bm \theta}\sum_{\bm x} \tilde P(\bm x) \\
    &= \frac{1}{Z} \sum_{\bm x} \nabla_{\bm \theta} \exp\left( \log\tilde P(\bm x)\right) \\
    &= \frac{1}{Z} \sum_{\bm x} \exp\left( \log\tilde P(\bm x)\right) \nabla_{\bm \theta} \log\tilde P(\bm x) \\
    &= \frac{1}{Z} \sum_{\bm x} \tilde P(\bm x) \nabla_{\bm \theta} \log\tilde P(\bm x) \\
    &= \sum_{\bm x} P(\bm x) \nabla_{\bm \theta} \log\tilde P(\bm x) \\
    &= \mathbb E_{\bm x \sim P}[\nabla_{\bm \theta}\log \tilde P(\bm x)]
\end{align*}
$$

Note that the introduction of $\tilde P(\bm x) = \exp\left(\log \tilde P(\bm x)\right)$ is valid since $\tilde P(\bm x) > 0$.

Therefore, both the positive phase and negative phase involve computing $\nabla_{\bm \theta} \log \tilde P(\bm x)$. But, the positive phase samples $\bm x$ from the data, that is, $\bm v \sim P_\text{data}(\bm v)$ and $\bm h \sim P_\text{model}(\bm h \mid \bm v)$, while the negative phase samples $\bm x$ from the model, that is, $\bm v, \bm h \sim P_\text{model}(\bm v, \bm h)$ (using Gibbs sampling).

## Training Restricted Boltzmann Machines with Contrastive Divergence

Sampling $\bm v$ and $\bm h$ from $P(\bm v, \bm h)$ for the negative phase in maximum likelihood learning requires Gibbs sampling, which means burning in a new set of Markov chains at every gradient step. A more efficient alternative to learning exists.

Let $P^{(k)}$ be the distribution of the model after $k$ full steps of Gibbs sampling. Suppose the initial sample (Step 1 of Gibbs sampling) is drawn from the data distribution, that is, $P^{(0)} = P_\text{data}$. The true distribution of the model $P_\text{model}$ is reached theoretically when the Markov chain converges after $\infty$ steps, that is, $P^{(\infty)} = P_\text{model}$. Since the goal of training is for the model to capture the underlying data distribution, we want $P^{(0)} = P^{(\infty)}$.

One way of representing a "distance"; between probability distributions $P$ and $Q$ is using the **Kullback-Leibler divergence**:
$$
    D_\text{KL}(P || Q) = \mathbb E_{ \mathrm{x} \sim P} \left[ \log P(x) - \log Q(x) \right]
$$
meaning "$Q$ differs from $P$ by $D_\text{KL}(P || Q)$".

**Contrastive divergence** trains the model by minimizing
$$
    D = D_\text{KL}( P^{(0)} || P^{(\infty)} ) - D_\text{KL}( P^{(1)} || P^{(\infty)} ).
$$
This works because $P^{(1)}$ is one step closer to the equilibrium distribution $P^{(\infty)}$ than $P^{(0)}$. Thus, $D_\text{KL}( P^{(0)} || P^{(\infty)} ) \geq D_\text{KL}( P^{(1)} || P^{(\infty)} )$, so $D$ is always non-negative. Also, if $D = 0$, then $P^{(0)} = P^{(1)}$, so the Markov chain has converged and $P^{(0)} = P^{(\infty)}$ (the model is perfect).

Minimizing $D$ requires $\nabla_{\bm \theta}D$. It can be shown that 
[(Hinton, 2002)](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)
$$
    \nabla_{\bm \theta} D = \mathbb E_{\mathbf x \sim P^{(0)}} \left[
        \nabla_{\bm \theta} \log \tilde P(\bm x)
    \right] - \mathbb E_{\mathbf x \sim P^{(1)}} \left[
        \nabla_{\bm \theta} \log \tilde P(\bm x)
    \right].
$$

So, the negative phase in maximum likelihood learning that required sampling from a burned-in  Markov chain now requires only one Gibbs sampling step before sampling: $\bm x \sim P^{(1)}$.

## Practical Application

See https://github.com/davidtranhq/pytorch-rbm for an RBM implemented in PyTorch. Training proceeds using contrastive divergence with momentum. Early stopping and L2 weight decay are used as regularizers.

![MNIST Loss](results/MNIST_loss.png)

![MNIST Generation](results/MNIST_generation.png)

![FashionMNIST Loss](results/FashionMNIST_loss.png)

![FashionMNIST Generation](results/FashionMNIST_generation.png)
