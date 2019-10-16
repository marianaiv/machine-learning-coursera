#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\machine-learning-coursera\Exercise3'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython
#%% [markdown]
# # Programming Exercise 3
# # Multi-class Classification and Neural Networks
# 
# In this exercise, you will implement one-vs-all logistic regression and neural networks to recognize handwritten digits.
#%%
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils
#%% [markdown]
# ## 1 Multi-class Classification
# 
# For this exercise, you will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes)
# on mail envelopes to recognizing amounts written on bank checks. This exercise will show you how the methods you have learned can be used for this classification task.
# 
# In the first part of the exercise, you will extend your previous implementation of logistic regression and apply it to one-vs-all classification.
# 
# ### 1.1 Dataset
# 
# You are given a data set in `ex3data1.mat` that contains 5000 training examples of handwritten digits (This is a subset of the [MNIST](http://yann.lecun.com/exdb/mnist) handwritten digit dataset). The `.mat` format means that that the data has been saved in a native Octave/MATLAB matrix format, instead of a text (ASCII) format like a csv-file. We use the `.mat` format here because this is the dataset provided in the MATLAB version of this assignment. Fortunately, python provides mechanisms to load MATLAB native format using the `loadmat` function within the `scipy.io` module. This function returns a python dictionary with keys containing the variable names within the `.mat` file. 
# 
# There are 5000 training examples in `ex3data1.mat`, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix `X`. This gives us a 5000 by 400 matrix `X` where every row is a training example for a handwritten digit image.
# 
# $$ X = \begin{bmatrix} - \: (x^{(1)})^T \: - \\ -\: (x^{(2)})^T \:- \\ \vdots \\ - \: (x^{(m)})^T \:-  \end{bmatrix} $$
# 
# The second part of the training set is a 5000-dimensional vector `y` that contains labels for the training set. 
#%%
# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size
#%% [markdown]
# ### 1.2 Visualizing the data
# 
# You will begin by visualizing a subset of the training set. In the following cell, the code randomly selects selects 100 rows from `X` and passes those rows to the `displayData` function. This function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together. We have provided the `displayData` function in the file `utils.py`. You are encouraged to examine the code to see how it works. Run the following cell to visualize the data.
#%%
# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)
#%% [markdown]
# ### 1.3 Vectorizing Logistic Regression
# 
# You will be using multiple one-vs-all logistic regression models to build a multi-class classifier. Since there are 10 classes, you will need to train 10 separate logistic regression classifiers. To make this training efficient, it is important to ensure that your code is well vectorized. In this section, you will implement a vectorized version of logistic regression that does not employ any `for` loops. You can use your code in the previous exercise as a starting point for this exercise. 
# 
# To test your vectorized logistic regression, we will use custom data as defined in the following cell.
#%%
# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3
#%% [markdown]
# <a id="section1"></a>
# #### 1.3.1 Vectorizing the cost function 
# 
# We will begin by writing a vectorized version of the cost function. Recall that in (unregularized) logistic regression, the cost function is
# 
# $$ J(\theta) = \frac{1}{m} \sum_{i=1}^m \left[ -y^{(i)} \log \left( h_\theta\left( x^{(i)} \right) \right) - \left(1 - y^{(i)} \right) \log \left(1 - h_\theta \left( x^{(i)} \right) \right) \right] $$
# 
# To compute each element in the summation, we have to compute $h_\theta(x^{(i)})$ for every example $i$, where $h_\theta(x^{(i)}) = g(\theta^T x^{(i)})$ and $g(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function. It turns out that we can compute this quickly for all our examples by using matrix multiplication. Let us define $X$ and $\theta$ as
# 
# $$ X = \begin{bmatrix} - \left( x^{(1)} \right)^T - \\ - \left( x^{(2)} \right)^T - \\ \vdots \\ - \left( x^{(m)} \right)^T - \end{bmatrix} \qquad \text{and} \qquad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix} $$
# 
# Then, by computing the matrix product $X\theta$, we have: 
# 
# $$ X\theta = \begin{bmatrix} - \left( x^{(1)} \right)^T\theta - \\ - \left( x^{(2)} \right)^T\theta - \\ \vdots \\ - \left( x^{(m)} \right)^T\theta - \end{bmatrix} = \begin{bmatrix} - \theta^T x^{(1)}  - \\ - \theta^T x^{(2)} - \\ \vdots \\ - \theta^T x^{(m)}  - \end{bmatrix} $$
# 
# In the last equality, we used the fact that $a^Tb = b^Ta$ if $a$ and $b$ are vectors. This allows us to compute the products $\theta^T x^{(i)}$ for all our examples $i$ in one line of code.
# 
# #### 1.3.2 Vectorizing the gradient
# 
# Recall that the gradient of the (unregularized) logistic regression cost is a vector where the $j^{th}$ element is defined as
# 
# $$ \frac{\partial J }{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( \left( h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x_j^{(i)} \right) $$
# 
# To vectorize this operation over the dataset, we start by writing out all the partial derivatives explicitly for all $\theta_j$,
# 
# $$
# \begin{align*}
# \begin{bmatrix} 
# \frac{\partial J}{\partial \theta_0} \\
# \frac{\partial J}{\partial \theta_1} \\
# \frac{\partial J}{\partial \theta_2} \\
# \vdots \\
# \frac{\partial J}{\partial \theta_n}
# \end{bmatrix} = &
# \frac{1}{m} \begin{bmatrix}
# \sum_{i=1}^m \left( \left(h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x_0^{(i)}\right) \\
# \sum_{i=1}^m \left( \left(h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x_1^{(i)}\right) \\
# \sum_{i=1}^m \left( \left(h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x_2^{(i)}\right) \\
# \vdots \\
# \sum_{i=1}^m \left( \left(h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x_n^{(i)}\right) \\
# \end{bmatrix} \\
# = & \frac{1}{m} \sum_{i=1}^m \left( \left(h_\theta\left(x^{(i)}\right) - y^{(i)} \right)x^{(i)}\right) \\
# = & \frac{1}{m} X^T \left( h_\theta(x) - y\right)
# \end{align*}
# $$
# 
# where
# 
# $$  h_\theta(x) - y = 
# \begin{bmatrix}
# h_\theta\left(x^{(1)}\right) - y^{(1)} \\
# h_\theta\left(x^{(2)}\right) - y^{(2)} \\
# \vdots \\
# h_\theta\left(x^{(m)}\right) - y^{(m)} 
# \end{bmatrix} $$
# 
# Note that $x^{(i)}$ is a vector, while $h_\theta\left(x^{(i)}\right) - y^{(i)}$  is a scalar (single number).
# The expression above allows us to compute all the partial derivatives
# without any loops. If you are comfortable with linear algebra, we encourage you to work through the matrix multiplications above to convince yourself that the vectorized version does the same computations. 
# 
# Your job is to write the unregularized cost function `lrCostFunction` which returns both the cost function $J(\theta)$ and its gradient $\frac{\partial J}{\partial \theta}$. Your implementation should use the strategy we presented above to calculate $\theta^T x^{(i)}$. You should also use a vectorized approach for the rest of the cost function. A fully vectorized version of `lrCostFunction` should not contain any loops.
#%%
def lrCostFunction(theta, X, y, lambda_):
    """
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta
    """
    #Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================
    h = utils.sigmoid(np.dot(X,theta))
    
    J = (1/m) * np.sum(
        np.dot(-y, np.log(h)) # first term
        -np.dot((1-y), np.log(1-h)) # second term
    )
    # agregamos el termino de regularizacion
    J += (lambda_/(2*m))*np.sum(theta[1:]*theta[1:])

    grad = (1/m)*np.dot(X.T,(h-y))  
    grad[1:] += (lambda_/m)*theta[1:] # because we don't add anything for j = 0
        
    # =============================================================
    return J, grad
#%% [markdown]
# #### 1.3.3 Vectorizing regularized logistic regression
# 
# After you have implemented vectorization for logistic regression, you will now
# add regularization to the cost function. Recall that for regularized logistic
# regression, the cost function is defined as
# 
# $$ J(\theta) = \frac{1}{m} \sum_{i=1}^m \left[ -y^{(i)} \log \left(h_\theta\left(x^{(i)} \right)\right) - \left( 1 - y^{(i)} \right) \log\left(1 - h_\theta \left(x^{(i)} \right) \right) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 $$
# 
# Note that you should not be regularizing $\theta_0$ which is used for the bias term.
# Correspondingly, the partial derivative of regularized logistic regression cost for $\theta_j$ is defined as
# 
# $$
# \begin{align*}
# & \frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right) x_j^{(i)}  & \text{for } j = 0 \\
# & \frac{\partial J(\theta)}{\partial \theta_0} = \left( \frac{1}{m} \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} \right) + \frac{\lambda}{m} \theta_j & \text{for } j  \ge 1
# \end{align*}
# $$
# 
# Now modify your code in lrCostFunction in the [**previous cell**](#lrCostFunction) to account for regularization.
#
# Once you finished your implementation, you can call the function `lrCostFunction` to test your solution using the following cell:

#%%
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]');