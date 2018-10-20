# Trading strategy optimization

## Problem setup
We are providing you with 5 engineered features $X \in \mathbb{R}^{n,5}$, and the corresponding market spread $s \in \mathbb{R}^n$ for a single asset. In addition to the explanatory power of the features, there are also autoregressive features that can be derived from the timeseries. You may wish to take this approach, but any timeseries features you wish to use need a 2 day lag to match the time horizons of the market settlements.

The goal is to learn a function $f(X)$ that returns a vector of trading volumes $v \in \mathbb{R}^{n}$ subject to the risk constraint of our worst loss, which for this exercise is 1000 dollars.  $v$ can take on both + values (short) and - values (long).

Formally, the problem becomes:

$\max{\sum_{i=1}^{n}}v_i s_i$

subject to 

-1000 <= $\min(vs)$

where $v=f(X)$



