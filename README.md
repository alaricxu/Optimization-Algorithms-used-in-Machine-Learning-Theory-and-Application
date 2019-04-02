# Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application

Inspired by **Convex Optimization**-Boyd and Vandenberghe, to get the book, click [Convex Optimization pdf version](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

See the update history [Commit Log](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Commit%20Log.txt)

### This repo will introduce regular optimization algorithms with its Python implementation

In the first part, Line Search Algorithm will be introduced. For Line Search definition and example, please review [Line Search Example](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Term%20Explanation-Line%20Search%20%26%20Trust%20Region.pdf)

#### Newton's method

In calculus, Newton's method refers to an iterative method for finding the roots of a differentiable function ![img](https://latex.codecogs.com/gif.latex?f), which are solutions to the equation ![Equation 1](https://latex.codecogs.com/gif.latex?f%28x%29%3D0). In optimization, Newton's method is applied to the derivative ![img](https://latex.codecogs.com/gif.latex?f%27) of a twice-differentiable function ![img](https://latex.codecogs.com/gif.latex?f) to find the roots of the derivative.

**Geometric interpretation**

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Newton_optimization_vs_grad_descent.svg/220px-Newton_optimization_vs_grad_descent.svg.png)

At each iteration, ![img](https://latex.codecogs.com/gif.latex?f%28x%29) is approximated by a quadratic function around ![img](https://latex.codecogs.com/gif.latex?x_%7Bn%27%7D) and a step is taken to the optimized value of the quadratic function (note in higher dimensions, this may lead to a saddle point).

**Higher dimensions**

Replace the derivative with the gradient ![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x%29), and denote the second derivative with the inverse of the Hessian matrix, ![img](https://latex.codecogs.com/gif.latex?Hf%28x%29), we have the iteratively update schema as<br> ![img](https://latex.codecogs.com/gif.latex?x_%7Bn&plus;1%7D%3Dx_n-%5BHf%28x_n%29%5D%5E%7B-1%7D%5Cnabla%20f%28x_n%29%2C%20n%3E0)<br>Usually, we add a step size <br>![img](https://latex.codecogs.com/gif.latex?x_%7Bn&plus;1%7D%3Dx_n-%5Cgamma%5BHf%28x_n%29%5D%5E%7B-1%7D%5Cnabla%20f%28x_n%29%2C%20n%3E0)<br>Normally, adding the step size (learning rate) to the update equation is to ensure that Wolfe conditions are satisfied at each step ![img](https://latex.codecogs.com/gif.latex?x_n%20%5Crightarrow%20x_%7Bn&plus;1%7D) of the iteration.

Where applicable, Newton's method converges much faster towards a local maximum or minimum than gradient descent. Each local minimum has a neighborhood N such that, if start with ![img](https://latex.codecogs.com/gif.latex?x_0%20%5Cin%20N), and if the Hessian is invertible and a Lipschitz continuous function [2] of x in the neighborhood, Newton's method with step size ![img](https://latex.codecogs.com/gif.latex?%5Cgamma%3D1) will converge quadratically. Note that if the Hessian is close to a non-invertible matrix, the inverted Hessian will be numerically unstable and the solution may diverge. Multiple metrics has been studied for this problem. For instance, the Quasi-Newton method [3].

We can view the whole process as calculating the vector ![img](https://latex.codecogs.com/gif.latex?%5CDelta%20x%3Dx_%7Bn&plus;1%7D-x_n)for the  equation system ![img](https://latex.codecogs.com/gif.latex?%5BHf%28x_n%29%5D%20%5CDelta%20x%3D%5Cnabla%20f%28x_n%29). To solve this linear equations system, iterative matrix factorization or approximation metrics may be applied.

#### Quasi-Newton method

As its name implies, Quasi-Newton method serves as an alternative to Newton's method, to find zeroes or local maxima and minima of functions. They can be used if the Jacobian or Hessian is unavailable or is too expensive to compute at every iteration.

**Search for zeros: root finding**

Newton's method to find zeroes of a function ![img](https://latex.codecogs.com/gif.latex?g) of multiple variables is given by ![img](https://latex.codecogs.com/gif.latex?x_%7Bn&plus;1%7D%3Dx_n-%5BJ_g%28x_n%29%5D%5E%7B-1%7Dg%28x_n%29), where ![img](https://latex.codecogs.com/gif.latex?%5BJ_g%28x_n%29%5D%5E%7B-1%7D)is the left inverse of the Jacobian matrix ![img](https://latex.codecogs.com/gif.latex?J_g%28x_n%29)of ![img](https://latex.codecogs.com/gif.latex?g) evaluated for ![img](https://latex.codecogs.com/gif.latex?x_%7Bn%7D). Strictly speaking, any method that replaces the exact Jacobian matrix with an approximation is a quasi-Newton method. Common approaches for find zeros are Broyden's "good" and "bad" methods, column-updating method, inverse column-updating method, quasi-Newton least squares method and the quasi-Newton inverse least squares method.

**Search for extrema: optimization**

If ![img](https://latex.codecogs.com/gif.latex?g)is gradient of ![img](https://latex.codecogs.com/gif.latex?f), then searching for the zeroes of the vector-valued function![img](https://latex.codecogs.com/gif.latex?g)corresponds to the search for the extrema of the scalar-valued function ![img](https://latex.codecogs.com/gif.latex?f); the Jacobian of ![img](https://latex.codecogs.com/gif.latex?g)now becomes the Hessian of ![img](https://latex.codecogs.com/gif.latex?f).

As in Newton's method, one uses a second-order approximation to find the minimum of a function ![img](https://latex.codecogs.com/gif.latex?f%28x%29). The Taylor series of ![img](https://latex.codecogs.com/gif.latex?f%28x%29) around an iterate is ![img](https://latex.codecogs.com/gif.latex?f%28x_k&plus;%5CDelta%20x%29%5Capprox%20f%28x_k%29&plus;%5Cnabla%20f%28x_k%29%5ET%20%5CDelta%20x&plus;%5Cfrac%7B1%7D%7B2%7D%20%5CDelta%20x%5ET%20B%20%5CDelta%20x) where (![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f)) is the gradient, and ![img](https://latex.codecogs.com/gif.latex?B)an approximation to the Hessian matrix. The gradient of this approximation is ![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x_k&plus;%5CDelta%20x%29%5Capprox%20%5Cnabla%20f%28x_k%29&plus;B%20%5CDelta%20x). Set this gradient to zero, we have the Newton step: ![img](https://latex.codecogs.com/gif.latex?%5CDelta%20x%3D-B%5E%7B-1%7D%20%5Cnabla%20f%28x_k%29). The Hessian approximation ![img](https://latex.codecogs.com/gif.latex?B)is chosen to satisfy ![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x_k&plus;%5CDelta%20x%29%3D%5Cnabla%20f%28x_k%29&plus;B%20%5CDelta%20x).

*The Update Process*

- ![img](https://latex.codecogs.com/gif.latex?%5CDelta%20x_k%3D-%5Calpha_k%20B_k%5E%7B-1%7D%20%5Cnabla%20f%28x_k%29) with ![img](https://latex.codecogs.com/gif.latex?%7B%5Calpha%7D)chosen to satisfy the Wolfe conditions
- Update use the equation: ![img](https://latex.codecogs.com/gif.latex?x_%7Bk&plus;1%7D%3Dx_k&plus;%5CDelta%20x_k)
- The gradient computed at the new point ![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x_%7Bk&plus;1%7D%29), and ![img](https://latex.codecogs.com/gif.latex?y_k%3D%20%5Cnabla%20f%28x_%7Bk&plus;1%7D%29%20-%5Cnabla%20f%28x_k%29)used to update the approximate Hessian ![img](https://latex.codecogs.com/gif.latex?%7BB_%7Bk&plus;1%7D%7D)

#### Gradient Descent

![_images/gradient_descent_demystified.png](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)

Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. It is based on the observation that if the multi-variable function **F(x)** is defined and differentiable in a neighborhood of a point ![img](https://latex.codecogs.com/gif.latex?a), then **F(x)** decreases fastest if one goes from ![img](https://latex.codecogs.com/gif.latex?a) in the direction of the negative gradient of **F(x)** at ![img](https://latex.codecogs.com/gif.latex?a), ![img](https://latex.codecogs.com/gif.latex?-%5Cnabla%20F%28a%29), and the update function is:<br>![img](https://latex.codecogs.com/gif.latex?%5Calpha_%7Bn&plus;1%7D%3D%5Calpha_n-%20%5Cgamma%20%5Cnabla%20F%28a%29)

Clearly, we can see that ![img](https://latex.codecogs.com/gif.latex?F%28n%29%20%5Cgeq%20F%28n&plus;1%29), this is a monotonic sequence. The optimized step size ![img](https://latex.codecogs.com/gif.latex?%5Cgamma_%7B%7D) is changing every step. With particular choice of ![img](https://latex.codecogs.com/gif.latex?%5Cgamma_%7B%7D)(by a line search that satisfies the Wolfe conditions or the Barzilai-Borwein method as ![img](https://latex.codecogs.com/gif.latex?%5Cgamma_n%3D%5Cfrac%7B%28x_n-x_%7Bn-1%7D%5ET%5B%5Cnabla%20F%28x_n%29-%5Cnabla%20F%28x_%7Bn-1%7D%29%5D%29%7D%7B%7C%7C%20%5Cnabla%20F%28x_n%29-%5Cnabla%20F%28x_%7Bn-1%7D%29%5E2%7C%7C%7D) ), the convergence will be guaranteed. When the function is convex, all local minima are also global minima, so the gradient descent can converge to the global solution.



#### Boosting Method, Gradient Boosting vs AdaBoost

See the detailed introduction in [Boosting Algorithm in Classification](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Thoughts%20on%20AdaBoost%20%26%20Gradient%20Boosting.pdf)




