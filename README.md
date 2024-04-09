# Non-concave distributionally robust stochastic control in a discrete time finite horizon setting

Ariel Neufeld and Julian Sester.

## Abstract
In this article we present a general framework for non-concave distributionally robust stochastic control problems in a  discrete time finite horizon setting. We do not impose any Markovian structure neither on the state process, nor on the actions taken by the agent. 
Our framework allows to consider a variety of different path-dependent ambiguity sets of probability measures comprising, as a natural example, the ambiguity set defined via Wasserstein-balls around path-dependent reference measures, as well as parametric classes of probability distributions. We establish a dynamic programming principle which allows to derive both optimal control and worst-case measure by solving recursively a sequence of one-step optimization problems.

As a concrete application, we study the robust hedging problem of a financial derivative under an asymmetric (and non-convex) loss function accounting for different preferences of sell- and buy side when it comes to the hedging of financial derivatives. As our entirely data-driven ambiguity set of probability measures, we consider Wasserstein-balls around the empirical measure derived from real financial data. We demonstrate that during adverse scenarios such as a financial crisis, our robust approach outperforms typical model-based hedging strategies such as the classical \emph{Delta-hedging} strategy as well as the hedging strategy obtained in the non-robust setting with respect to the empirical measure and therefore overcomes the problem of model misspecification in such critical periods. 


Notebooks:
The file 'Hedging' contains the python notebook related to the example from Section 5.



## License
MIT License

Copyright (c) 2021 Julian Sester

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
