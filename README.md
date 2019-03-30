# Optimization-Algorithms-used-in-Machine-Learning-Theory-and-Application

### This repo will introduce regular optimization algorithms with its Python implementation

#### Newton's method

In calculus, Newton's method refers to an iterative method for finding the roots of a differentiable function ![img](https://latex.codecogs.com/gif.latex?f), which are solutions to the equation ![Equation 1](https://latex.codecogs.com/gif.latex?f%28x%29%3D0) [1]. In optimization, Newton's method is applied to the derivative ![img](https://latex.codecogs.com/gif.latex?f%27) of a twice-differentiable function ![img](https://latex.codecogs.com/gif.latex?f) to find the roots of the derivative.

**Geometric interpretation**

At each iteration, ![img](https://latex.codecogs.com/gif.latex?f%28x%29) is approximated by a quadratic function around ![img](https://latex.codecogs.com/gif.latex?x_%7Bn%27%7D) and a step is taken to the optimized value of the quadratic function (note in higher dimensions, this may lead to a saddle point).

**Higher dimensions**

Replace the derivative with the gradient ![img](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x%29), and denote the second derivative with the inverse of the Hessian matrix, ![img](https://latex.codecogs.com/gif.latex?Hf%28x%29), we have the iteratively update schema as<br> ![img](https://latex.codecogs.com/gif.latex?x_%7Bn&plus;1%7D%3Dx_n-%5BHf%28x_n%29%5D%5E%7B-1%7D%5Cnabla%20f%28x_n%29%2C%20n%3E0)<br>Usually, we add a step size <br>![img](https://latex.codecogs.com/gif.latex?x_%7Bn&plus;1%7D%3Dx_n-%5Cgamma%5BHf%28x_n%29%5D%5E%7B-1%7D%5Cnabla%20f%28x_n%29%2C%20n%3E0)<br>Where applicable, Newton's method converges much faster towards a local maximum or minimum than gradient descent.

We can view the whole process as calculating the vector ![img](https://latex.codecogs.com/gif.latex?%5CDelta%20x%3Dx_%7Bn&plus;1%7D-x_n)for the  equation system ![img](https://latex.codecogs.com/gif.latex?%5BHf%28x_n%29%5D%20%5CDelta%20x%3D%5Cnabla%20f%28x_n%29).





**Reference** 

[1] <https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>