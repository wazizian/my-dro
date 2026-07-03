The goal is to do robust hedging

Model: Black scholes or dupire for price trajectories and sample a few to simulate having some empirircal trajectories

Follow this setting:
Consider a discrete-time financial market with finite time horizon $T$ and trading dates $0=t_{0}<\ldots<t_{n}=T$. Let $I=(I_{k})_{k=0,\dots, n}$ be an $\mathbb{R}^{r}$ valued process that generates the filtration $\mathbb{F}=\left(\mathcal{F}_{k}\right)_{k=0}^{n}$
, where $\mathcal{F}_{k}$ means all of the information known up to time $t_{k}$. Let $Z$ be a $\mathcal{F}_{T}$ measurable liability that needs to be hedged. We choose the $d$-hedging instruments with maximal maturity time $T$, $V=\left(V_{k}\right)_{k=0}^{n}$ being $\mathbb{R}^{d}$-valued $\mathbb{F}$ adapted, with $V_{k}^{i}$ denoting the mid-price of the $i$-th asset at time $t_{k}$.
Let $\delta=\left(\delta_{k}\right)_{k=0}^{n-1}$ be an $\mathbb{F}$-adapted with $\delta_{k}=\left(\delta_{k}^{1}, \cdots, \delta_{k}^{d}\right)$, with $\delta_{k}^{i}$ denoting agent's holding of the $i-$th asset at time $t_{k}$. We denote $R=\left(R_{k}\right)_{k=0}^{n}$ to be $\mathbb{F}$-adapted risk free rate at time $t_{k}$.

For simplicity, the all instruments above are settled in one reference currency. Alos: for now the info I are just the prices


Goal: we aim to optimize the problem below numerically
\begin{equation}\label{problem}
    \inf _{\delta \in M} \sup _{\pi \in A(\hat{\pi}, \varepsilon,\mathcal{V} )} \quad \mathbb{E}_{\pi}\left[\text{Loss}^{\text {dist }}(\delta)\right]
\end{equation}

where Loss = -PnL

$$
\begin{gathered}
\text{PnL}^{\text {dist }}(\delta)=\sum_{k=0}^{n-1} \delta_k^{\top}\left(\frac{V_{k+1}}{B_{k+1}}-\frac{V_k}{B_k}\right)-\sum_{k=0}^{n-1} C_k\left(\delta_k-\delta_{k-1}\right)-\frac{Z}{B_T} \\
B_k=\prod_{j=0}^{k-1}\left(1+R_j\right)
\end{gathered}
$$

The stragey delta will be a small linear model / MLP


Method 1: Deep hedging

Just optimize the objective above


Method 2: Robust hedging

Will implement later

Tools:
- Jax
- Optax
- Equinox
- Use tyro for config in a python file
- uv
(Note: assume everything is a pytree)

