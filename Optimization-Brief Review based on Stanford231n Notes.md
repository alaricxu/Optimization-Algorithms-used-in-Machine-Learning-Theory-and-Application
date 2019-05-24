Key Components in Context of the Image Classification Task:

- A (parameterized) score function mapping the raw image pixels to class scores (e.g. a linear function)
- A loss function that measured the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data.

The goal of optimization is to find **W** that minimizes the loss function.

##### Random search

```python
bestloss = float("inf")
for num in range(1000):
    W = np.random.randn(10, 3073) * 0.0001
    loss = L(X_train, Y_train, W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print('in attempt %d the loss was %f, best %f' % (num, loss, bestloss))
    

scores = Wbest,dot(Xte_cols)

# find the index with max score in each column
Yte_predict = np.argmax(scores, axis=0)
# and calculate accuracy
np.mean(Yte_predict == Yte)
```

##### Core idea: iterative refinement

Random Local Search

```python
W = np.random.randn(10, 3073) * 0.001
bestloss = float('inf')
for i in range(1000):
    step_size = 0.0001
    Wtry = W + np.random.randn(10, 3073) * step_size
    loss = L(Xtr_cols, Ytr, Wtry)
    if loss < bestloss:
        W = Wtry
        bestloss = loss
    print('iter %d loss is %f', %(i, bestloss))

```

##### Following the Gradient

$\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$ 

When the functions of interest take a vector of numbers instead of a single number, we call the derivatives partial derivatives, and the gradient is simply the vector of partial derivatives in each dimension.

Compute the gradient numerically with finite differences

```python
def eval_numerical_gradient(f, x):
    # a naive implementation of numerical gradient of f at x
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001
    
    # iterate over all indexes in x
    it = np.nditer(x, flags = ['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        # increment by h
        fxh = f(x)
        # evaluate f(x + h)
        x[ix] = old_value
        # restore to previous value (very important)
        
        # compute the partial derivative
        grad[ix] = (fxh - fx) / h
        # the slope
        it.iternext()
        # step to next dimension
    return grad
```

Following the gradient formula we gave above, the code above iterates over all dimensions one by one, makes a small change *h* along the dimension and calculates the partial derivative of the loss function along that dimension by seeing how much the function changed. The variable *grad* holds the full gradient in the end.

In practice, it often works better to compute the numeric gradient using the centered difference formula: $[f(x+h) - f(x-h)/2h]$

Compute the gradient analytically with Calculus

Analytically using Calculus allows us to derive a direct formula for the gradient (no approximations) that is also very fast to compute. However, unlike the numerical gradient, it can be more error prone to implement, which is why in practice it is very common to compute the analytic gradient and compare it to the numerical gradient to check the correctness of the implementation, which is called a gradient check.

$L_i = \sum_{j \neq y_i} [max (0, w_j^T x_i - w_{y_i}^T x_i + \Delta)]$

We can differentiate the function with respect to the weights. For example, taking the gradient with respect to $w_{ui}$ we obtain:

$\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i$

where 1 is the indicator function that is one if the condition inside is true or zero otherwise. 

This is the gradient only with respect to the row of **W** that corresponds to the correct class. For the other rows where $j \neq y_i$ the gradient is:

$\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i​$

Once you derive the expression for the gradient it is straight-forward to implement the expressions and use them to perform the gradient update.

Gradient Descent

We can compute the gradient of the loss function, the procedure of repeatedly evaluating the gradient and then performing a parameter update is called *Gradient Descent*. Its vanilla version looks as follows:

```python
# vanilla gradient descent
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weights_grad
    # perform parameter update
```

Mini-batch gradient descent

```python
# vanilla minibatch gradient descent
while True:
    data_batch = sample_training_data(data, 256)
    # sample 256 samples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    # perform parameter update
    weights += -step_size * weights_grad
```

