# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\machine-learning-coursera\Exercise2'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # Programming Exercise 2: Logistic Regression
 
#%%
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1 Logistic Regression
# 
# In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university. Suppose that you are the administrator of a university department and
# you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions
# decision. Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams. 
# 
# The following cell will load the data and corresponding labels:

#%%
# Load data
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

#%%
#%% [markdown]
# ### 1.1 Visualizing the data
# 
# Before starting to implement any learning algorithm, it is always good to visualize the data if possible. We  display the data on a 2-dimensional plot by calling the function `plotData`. You will now complete the code in `plotData` so that it displays a figure where the axes are the two exam scores, and the positive and negative examples are shown with different markers.

#%%
def plotData(X, y):
    """
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """
    # Create New Figure
    fig = pyplot.figure()

    # ====================== YOUR CODE HERE ======================
    pos = y == 1
    neg = y == 0
    pyplot.plot(X[pos, 0], X[pos, 1], 'b*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'o', mfc='y', ms=8, mec='k', mew=1)
    
    # ============================================================


#%%
# Now, we call the implemented function to display the loaded data:


#%%
plotData(X, y)
# add axes labels
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
pass

#%% [markdown]
# <a id="section1"></a>
# ### 1.2 Implementation
# 
# #### 1.2.1 Warmup exercise: sigmoid function
# 
# Before you start with the actual cost function, recall that the logistic regression hypothesis is defined as:
# 
# $$ h_\theta(x) = g(\theta^T x)$$
# 
# where function $g$ is the sigmoid function. The sigmoid function is defined as: 
# 
# $$g(z) = \frac{1}{1+e^{-z}}$$.
# 
# Your first step is to implement this function `sigmoid` so it can be
# called by the rest of your program. When you are finished, try testing a few
# values by calling `sigmoid(x)` in a new cell. For large positive values of `x`, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. Evaluating `sigmoid(0)` should give you exactly 0.5.
#%%
def sigmoid(z):
    """
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = 1/(1+np.exp(-z))

    # =============================================================
    return g

#%% [markdown]
# The following cell evaluates the sigmoid function at `z=0`. You should get a value of 0.5. You can also try different values for `z` to experiment with the sigmoid function.

#%%
# Test the implementation of sigmoid function here
z = 0
g = sigmoid(z)

print('g(', z, ') = ', g)

#%% [markdown]
# <a id="section2"></a>
# #### 1.2.2 Cost function and gradient
# 
# Now you will implement the cost function and gradient for logistic regression. Before proceeding we add the intercept term to X. 

#%%
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

#%% [markdown]
# Now, complete the code for the function `costFunction` to return the cost and gradient. Recall that the cost function in logistic regression is
# 
# $$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \log\left(h_\theta\left( x^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - h_\theta\left( x^{(i)} \right) \right) \right]$$
# 
# and the gradient of the cost is a vector of the same length as $\theta$ where the $j^{th}$
# element (for $j = 0, 1, \cdots , n$) is defined as follows:
# 
# $$ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta \left( x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} $$

#%%
def costFunction(theta, X, y):
    """    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    
    h = sigmoid(np.dot(theta,np.transpose(X)))

    J = (1/m)*np.sum((np.dot(-y,np.log(h)))-(np.dot((1-y),np.log(1-h))))

    for i in range(0,grad.size):
        grad[i] = (1/m)*np.sum(np.dot((h-y),X[:,i]))
    
    # =============================================================
    return J, grad

#%% [markdown]
# Once you are done call your `costFunction` using two test cases for  $\theta$ by executing the next cell.

#%%
# Initialize fitting parameters
initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

#%%
