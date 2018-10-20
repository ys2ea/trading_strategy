# Trading strategy optimization

## Problem setup
We have 5 engineered features as a vector X for each hour, and the corresponding market spread s.

In addition to the explanatory power of the features, there are also autoregressive features that can be derived from the timeseries and the spread is related to the features of the past 48 hours.

The goal is to learn a function f(X) that returns a vector of trading volumes v subject to the risk constraint of our worst loss, which for this exercise is 1000 dollars.  v can take on both + values 

Formally, the problem becomes:

![](https://latex.codecogs.com/gif.latex?%24%5Cmax%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Dv_i%20s_i%24)

subject to

![](https://latex.codecogs.com/gif.latex?%24%5Cmin%28vs%29%24%20%3E%3D%20-1000)


## Summary of method
We use a recurrent neural network to study the spread, since RNN is sensitive to the sequence of the data and take account of the autoregressive feature. The correlation length of the RNN is set to be 48 to account for the lag.

Inorder to maximize the profit, we use a simple reinforcement learning procedure based on policy gradient where the function to be minimized is set to 

![](https://latex.codecogs.com/gif.latex?%24-%5Csum_i%20v_i%20s_i%24)

A constraint is added by a penalty term 
![](https://latex.codecogs.com/gif.latex?e%5E%7B-%28%5Cmin%28vs%29&plus;1000%29%7D)
that heavily disfavors losses that are more than 1000.

L2 regulation term and dropout layer is also used to prevent overfitting.



