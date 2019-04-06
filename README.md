# Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application

Inspired by **Convex Optimization**-Boyd and Vandenberghe, to get the book, click [Convex Optimization pdf version](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

See the update history [Commit Log](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Commit%20Log.txt)

### This repo will introduce regular optimization algorithms with its Python implementation



**At the very beginning, Line Search Algorithm will be introduced. For Line Search definition and example, please review [Line Search Example](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Term%20Explanation-Line%20Search%20%26%20Trust%20Region.pdf)**

### Part 1 Gradient-free Optimization

to be continue...



### Part 2 Gradient-based Optimization

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

**Problem with Gradient Descent: Ravines and Saddle Points**

Gradient descent becomes difficult when the function being optimized looks locally like a ravine or a saddle.

![âravine gradient descentâçå¾çæç´¢ç"æ](https://www.jeremyjordan.me/content/images/2017/11/minmaxsaddle.png)

When in the first two scenarios, gradient descent can easily locate the global optimum. However, when in the last scenario, the  gradient descent is likely to be trapped into local optimum, while local optimum is not necessarily global optimum.

#### Practical Improvement to Gradient Descent: Momentum, Stochastic Gradient Descent, Batch Gradient Descent, Mini-Batch Gradient Descent

#### Stochastic Gradient Descent

Standard Gradient Descent evaluates the sum-gradient, which may require expensive evaluations of the gradients from all sums and functions. When the training set is enormous and no simple formulas exist, evaluating the sums of gradients becomes very expensive, because evaluating the gradients requires evaluating all the summand functions' gradients. In stochastic ("on-line") gradient descent, the true gradient of Q(w) is approximated by a gradient at a single example.

![img](https://latex.codecogs.com/gif.latex?w%20%3A%3D%20w%20-%20%5Ceta%20%5Cnabla%20Q_i%28w%29)

We compute an estimate or approximation to this direction. The most simple way is to just look at one training example (subset of training examples) and compute the direction to move only on this approximation. It is called as Stochastic because the approximate direction that is computed at every step can be though of a random variable of a stochastic process. Stochastic method converges much faster compared to standard, but the error function is not as well minimized as in the case of the latter. Often in most cases, the close approximation that you get in stochastic method for the parameter values are enough because they reach the optimal values and keep oscillating there.

#### Minibatch Gradient Descent

A compromise between the two forms called "mini-batches" computes the gradient against more than one training examples at each step. This can perform significantly better than true stochastic gradient descent, because the code can make use of vectorization libraries rather than computing each step separately. It may also result in smoother convergence, as the gradient computed at each step uses more training examples.

#### Gradient Descent Variants

#### Momentum

Gradient Descent struggles navigating ravines, areas where the surface curves much more steeply in one dimension than in another. Once fallen into ravine, Gradient Descent oscillates across the slopes of the ravine, without making much more progress towards the local optimum. Momentum techniques accelerates Gradient Descent in the relevant direction and lessens oscillations. In the illustrations below, the left one is vanilla Gradient Descent and the right is Gradient Descent with Momentum

![img](https://1.bp.blogspot.com/-eVq8WSmhxfE/V1K_MNMTjjI/AAAAAAAAFUo/Si6N7fkGErQO3aRitHsY_xTDyABDORU_gCLcB/s400/momentum.png)

When the Momentum technique is applied, the fraction of the update vector of the past time step is added to the current update vector:

![img](https://latex.codecogs.com/gif.latex?v_t%3D%5Cgamma%20v_%7Bt-1%7D%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta_%7Bt-1%7D%29)

![img](https://latex.codecogs.com/gif.latex?%5Ctheta_t%20%3D%20%5Ctheta_%7Bt-1%7D%20&plus;%20v_t)

The momentum parameter is usually set to  0.9
The idea behind using momentum accelerating speed of the ball as it rolls down the hill, until it reaches its terminal velocity if there is air resistance, that is our parameter ![img](https://latex.codecogs.com/gif.latex?%5Cgamma). Similarly, the momentum increases updates for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions gaining faster convergence while reducing oscillation.

#### Nesterov Accelerated Gradient

The standard momentum method first computes the gradient at the current location and then takes a big jump in the direction of the updated accumulated gradient. The Nesterov Accelerated Gradient (NAG) looks ahead by calculating the gradient not by our current parameters but by approximating future position of our parameters. In the following illustration, instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this "looked-ahead" position.

![img](https://1.bp.blogspot.com/-eX6i_P4d50A/V1P2nPYQBaI/AAAAAAAAFdo/3P9slgH5vWE36pgIfYnYiI0cRW8J5HroQCKgB/s640/nesterov.jpeg)

The formula for Nesterov accelerated gradient is as following with momentum parameter set to 0.9.

![img](https://latex.codecogs.com/gif.latex?v_t%3D%5Cgamma_%7Bt-1%7D%20v_%7Bt-1%7D%20-%20%5Ceta_%7Bt-1%7D%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta_%7Bt-1%7D%20&plus;%5Cgamma_%7Bt-1%7D%20v_%7Bt-1%7D%29)

![img](https://latex.codecogs.com/gif.latex?%5Ctheta_t%20%3D%20%5Ctheta_%7Bt-1%7D%20&plus;%20v_t)

It enjoys stronger theoretical converge guarantees for convex functions and in practice it also consistently works slightly better than standard momentum.

#### Adagrad

All previous approaches we've discussed so far manipulated the learning rate globally and equally for all parameters. Adagrad is a well-suited algorithm for dealing with sparse data - it edits the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. Adagrad uses a different learning rate for every parameter ![img](https://latex.codecogs.com/gif.latex?%5Ctheta)at each step, and not an update for all parameters at once and given by:

![img](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bt&plus;1%7D%3D%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%20%7BG_t&plus;%5Cepsilon%7D%7D%20%5Cbigodot%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29)

where ![img](https://latex.codecogs.com/gif.latex?%5Cbigodot)is an element-wise multiplication, ![img](https://latex.codecogs.com/gif.latex?%5Cepsilon)is a smoothing term that avoids division by zero (usually on the order of 1e-8), ![img](https://latex.codecogs.com/gif.latex?G_t)is a diagonal matrix of sum of the squares of the past gradients -  ![img](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29%5E2)

One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate. Most implementations use a default value of 0.01 and leave it at that.

#### Adadelta

Adadelta is an improvement over Adagrad which reduces its aggressiveness and monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size **w**. 

With Adadelta, we do not even need to set a default learning rate.

#### RMSprop

RMSprop also tries to overcome the diminishing learning rates of Adagrad and works similarly to Adadelta as following:

![img](https://latex.codecogs.com/gif.latex?E%5B%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta%29%5E2%5D_t%20%3D%20%5Cgamma%20E%5B%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29%5E2%5D_%7Bt-1%7D%20&plus;%20%281-%5Cgamma%29%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29_t%5E2%20%5Ctheta_%7Bt&plus;1%7D%20%3D%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BE%5B%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta%29%5E2%5D_t%20&plus;%5Cepsilon%5D%7D%7D%20%5Cnabla_%7B%5Ctheta%7D%20J%28%5Ctheta%29_t)

where E is a running average. RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients. Momentum rate is usually set to 0.9, while a good default value for the learning rate is 0.001.

#### Boosting Method, Gradient Boosting vs AdaBoost

See the detailed introduction in [Boosting Algorithm in Classification](https://github.com/alaricxu/Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application/blob/master/Thoughts%20on%20AdaBoost%20%26%20Gradient%20Boosting.pdf)






