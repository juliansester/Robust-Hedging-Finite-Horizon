# Non-Concave Stochastic Optimal Control in Finite Discrete Time Under Model Uncertainty

Ariel Neufeld and Julian Sester

## Abstract
In this article, we present a general framework for non-concave robust stochastic control problems under model uncertainty in a discrete-time finite horizon setting. 

Our framework allows for various path-dependent ambiguity sets of probability measures. This includes ambiguity sets defined via Wasserstein-balls around path-dependent reference measures with path-dependent radii, as well as parametric classes of probability distributions. 

We establish a dynamic programming principle, enabling us to determine both the optimal control and worst-case measure by solving a sequence of one-step optimization problems recursively. Additionally, we derive upper bounds for the difference in values between the robust and non-robust stochastic control problems in cases of Wasserstein uncertainty and parameter uncertainty.

### Application: Robust Hedging of Financial Derivatives
As a concrete example, we explore the robust hedging of financial derivatives under an asymmetric and non-convex loss function. This accounts for the different preferences of sell- and buy-side parties in hedging financial derivatives.

For our entirely data-driven ambiguity set of probability measures, we use Wasserstein-balls around the empirical measure obtained from real financial data. We demonstrate that our robust approach outperforms classical model-based hedging strategies, including Delta-hedging and empirical measure-based hedging, particularly in adverse scenarios such as financial crises. This ensures better performance and mitigates the risk of model misspecification during critical periods.

## Notebooks

This repository includes several Python notebooks for calculations and result visualization:

### **One-Dimensional Case**
- **`Prospect_Calculations.ipynb`** – Performs calculations for the one-dimensional case.
- **`Prospect_Calculations_adaptive.ipynb`** – Implements robust adaptive calculations for the one-dimensional case.
- **`Prospect_results.ipynb`** – Displays the results for the one-dimensional case.

### **Multi-Dimensional Case**
- **`Prospect_Calculations_multidim.ipynb`** – Performs calculations for the multi-dimensional case.
- **`Prospect_Calculations_multidim_adaptive.ipynb`** – Implements robust adaptive calculations for the multi-dimensional case.
- **`Prospect_results_multi_dim.ipynb`** – Displays the results for the multi-dimensional case.

## Supporting Python Files

These files contain functions necessary for performing calculations and evaluating results:

### **Adaptive Calculations**
- **`functions_adaptive.py`** – Implements functions for the adaptive case.

### **Multi-Dimensional Calculations**
- **`functions_multidim.py`** – Implements functions for the multi-dimensional case.
- **`functions_multidim_adaptive.py`** – Implements functions for the multi-dimensional adaptive case.

### **Optimization and Results**
- **`optimization_functions.py`** – Implements functions for optimization in the one-dimensional case.
- **`results_functions.py`** – Implements functions to display and evaluate results.

## License

MIT License  

Copyright (c) 2025 Julian Sester  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.