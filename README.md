## Contributions to DiffeoMorph Manuscript
### Benjamin Fefferman

Please also see below for the causal inference algorithm I created for this project:

\section{Inferring Causal Morphogens - Multivariate Granger Causality for Geometric Structure - (Benjamin Fefferman)}

I sought to identify which features (genes) were causally implicated in changes in geometric structure between consecutive timesteps. To do this, I employed multivariate Granger causality on the Wasserstein distance between consecutive point clouds. This algorithm builds upon Granger causal inference on timeseries and DAGs by adding the Wasserstein structural objective, by adding a step for attention-based aggregation, and by fitting reduced models within a parallelized framework. For each time step, contextualized embeddings are computed for all cells, and averaged to produce a single global embedding. Then, for all steps beyond the first, the Earth Mover's (Wasserstein) distance is computed between point clouds at the current and previous timestep. A multivariate, time-lagged regression is then performed, for Wasserstein distance vs. cellular gene expression, between a ``full'' model comprised of all genes (features), and a ``reduced'' model in exclusion of a single gene. An $F$-statistic is used to compare the sum of squared residuals between the full and reduced models, in which case a $p$-value less than $\alpha$ implies that gene $j$'s expression ``Granger-causes'' alterations in structure at timestep $t$, within a sliding window (time lags) of $\ell={1,\dots,L}$ up to that timestep.

\begin{algorithm}
\renewcommand{\thealgorithm}{}
\caption{Multivariate Granger Causality, Geometric Structure}
\begin{algorithmic}[1]
\small
\State $\text{Causal\_Morphogens}\leftarrow[\;]$
\For{$t = 1$ to $T$}
  \State $\tilde{X}(t) \gets \text{Attention}(X(t))$\Comment{$X(t)$: $N \text{ cells} \times k \text{ genes}$ matr., time $t$ $\tilde{X}(t)$: Context. embeddings, time $t$}
  \State $\tilde{E}(t) \gets \frac{1}{N}\sum_{i=1}^N\tilde{X}_{[i,:]}(t)$\Comment{Aggregation (summation) of context. embeddings $\forall$ cell at time $t$}
  \If{$t > L$} \Comment{For timesteps beyond extent of time lag at $t>L$}
    \State $D(t) \gets \text{EMD}(P_t, P_{t-1})$ \Comment{$D(t)$: Earth Mov. Dist. (EMD), pt. cl. $P_t$, $P_{t-1}$, time $t$}
    \State $D(t) = \sum_{\ell=1}^L \tilde{E}(t-\ell)W^{(1)}_{\ell}+\epsilon_1(t)$ \Comment{$\tilde{E}(t)$: Full model, $W^{1}_{\ell}$: Coeff.'s, full model (1), lag $\ell$, $\epsilon_1(t)$: Residuals}
    \ParFor{$j = 1$ to $k$}
        \State $R(t)\gets \tilde{E}(t)_{[:,-j]}$\Comment{$R(t)$: Reduced model, excluding the $j$-th feature (gene)}
        \State $D(t) = \sum_{\ell=1}^L R(t-\ell)W^{(2)}_{\ell} + \epsilon_2(t)$ \Comment{$W^{(1)}_{\ell}$: Coeff.'s, reduced model (2), lag $\ell$, $\epsilon_2(t)$: Residuals}
        \State $F_{stat_{i\leftarrow j}} \gets F_{stat}(\text{RSS}_1, \text{RSS}_2)$\Comment{$F_{stat_{i\leftarrow j}}$: F-statistic btw. sum of sq. residuals $\text{RSS}_1$, $\text{RSS}_2$}
        \If{$p\text{-value}(F_{stat_{i \leftarrow j}}) < \alpha$}\Comment{$\alpha$: Significance threshold for $F$-test}
            \State $\text{Causal\_Morphogens.append($j$)}$
        \Else
            \State $\text{Causal\_Morphogens.append(\verb|NaN|)}$
        \EndIf
    \EndParFor
  \EndIf
\EndFor

\end{algorithmic}

\end{algorithm}
(Lutkepohl \textit{et al.}, 2005), (Singh \textit{et al.}, 2022), (Blelloch \textit{et al.}, 2012)
