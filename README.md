# Trading strategy optimization

## Problem setup
We have 5 engineered features as a vector X for each hour, and the corresponding market spread s.

In addition to the explanatory power of the features, there are also autoregressive features that can be derived from the timeseries and the spread is related to the features of the past 48 hours.

The goal is to learn a function f(X) that returns a vector of trading volumes v subject to the risk constraint of our worst loss, which for this exercise is 1000 dollars.  $v$ can take on both + values ($

Formally, the problem becomes:

![](https://latex.codecogs.com/gif.latex?%24%5Cmax%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Dv_i%20s_i%24)

subject to

![](https://latex.codecogs.com/gif.latex?%24%5Cmin%28vs%29%24%20%3E%3D%20-1000)






