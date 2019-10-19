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
# To compute each element in the summation, we have to compute $h_\theta(x^{(i)})$ for every example $i$, where $h_\theta(x^{(i)}) = g(\theta^T x^{(i)})$ and $g(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function. It turns out that we can compute this quickly for all our examples by using matrix multiplication. By computing the matrix product $X\theta$, we have: 
# 
# $$ X\theta = \begin{bmatrix} - \left( x^{(1)} \right)^T\theta - \\ - \left( x^{(2)} \right)^T\theta - \\ \vdots \\ - \left( x^{(m)} \right)^T\theta - \end{bmatrix} = \begin{bmatrix} - \theta^T x^{(1)}  - \\ - \theta^T x^{(2)} - \\ \vdots \\ - \theta^T x^{(m)}  - \end{bmatrix} $$
# 
# This allows us to compute the products $\theta^T x^{(i)}$ for all our examples $i$ in one line of code.
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
# Your job is to write the unregularized cost function `lrCostFunction` which returns both the cost function $J(\theta)$ and its gradient $\frac{\partial J}{\partial \theta}$. A fully vectorized version of `lrCostFunction` should not contain any loops.
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
    # add the term for regularization
    J += (lambda_/(2*m))*np.sum(theta[1:]*theta[1:])

    grad = (1/m)*np.dot(X.T,(h-y))  

    # add regularization
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
# Once you finished your implementation, you can call the function `lrCostFunction` to test your solution:
#%%
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]');
#%% [markdown]
# <a id="section2"></a>
# ### 1.4 One-vs-all Classification
# 
# In this part of the exercise, you will implement one-vs-all classification by training multiple regularized logistic regression classifiers, one for each of the $K$ classes in our dataset. In the handwritten digits dataset, $K = 10$, but your code should work for any value of $K$. 
# 
# You should now complete the code for the function `oneVsAll` below, to train one classifier for each class. In particular, your code should return all the classifier parameters in a matrix $\theta \in \mathbb{R}^{K \times (N +1)}$, where each row of $\theta$ corresponds to the learned logistic regression parameters for one class. You can do this with a “for”-loop from $0$ to $K-1$, training each classifier independently.
# 
# Note that the `y` argument to this function is a vector of labels from 0 to 9. When training the classifier for class $k \in \{0, ..., K-1\}$, you will want a K-dimensional vector of labels $y$, where $y_j \in 0, 1$ indicates whether the $j^{th}$ training instance belongs to class $k$ $(y_j = 1)$, or if it belongs to a different
# class $(y_j = 0)$. You may find logical arrays helpful for this task. 
# 
# Furthermore, you will be using scipy's `optimize.minimize` for this exercise. 
# <a id="oneVsAll"></a>
#%%
def oneVsAll(X, y, num_labels, lambda_):
    """
    Instructions
    ------------
    You should complete the following code to train `num_labels`
    logistic regression classifiers with regularization parameter `lambda_`. 
    
    """
    # Some useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    # Set initial theta
    initial_theta = np.zeros(X.shape[1])
    # Set options for minimize
    options = {'maxiter': 50}

    # Minimize theta for each class 
    for ii in range(0,num_labels):
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y==ii), lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options)
        all_theta[ii,:]=res.x
    # ============================================================
    return all_theta
#%% [markdown]
# After you have completed the code for `oneVsAll`, the following cell will use your implementation to train a multi-class classifier. 
#%%
lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)
#%% [markdown]
# <a id="section3"></a>
# #### 1.4.1 One-vs-all Prediction
# 
# After training your one-vs-all classifier, you can now use it to predict the digit contained in a given image. For each input, you should compute the “probability” that it belongs to each class using the trained logistic regression classifiers. Your one-vs-all prediction function will pick the class for which the corresponding logistic regression classifier outputs the highest probability and return the class label (0, 1, ..., K-1) as the prediction for the input example. You should now complete the code in the function `predictOneVsAll` to use the one-vs-all classifier for making predictions. 
# <a id="predictOneVsAll"></a>
#%%
def predictOneVsAll(all_theta, X):
    """     
    Instructions
    ------------
    Complete the following code to make predictions using your learned logistic
    regression parameters (one-vs-all). You should set p to a vector of predictions
    (from 0 to num_labels-1).

    """
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================

    ClassP = utils.sigmoid(np.dot(X,all_theta.T))
    p = np.argmax(ClassP, axis=1)

    # ============================================================
    return p
#%% [markdown]
# Once you are done, call your `predictOneVsAll` function using the learned value of $\theta$. You should see that the training set accuracy is about 95.1% (i.e., it classifies 95.1% of the examples in the training set correctly).
#%%
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))
#%% [markdown]
# ## 2 Neural Networks
# 
# In the previous part of this exercise, you implemented multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot form more complex hypotheses as it is only a linear classifier (You could add more features - such as polynomial features - to logistic regression, but that can be very expensive to train).
# 
# In this part of the exercise, you will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. For this week, you will be using parameters from a neural network that we have already trained. Your goal is to implement the feedforward propagation algorithm to use our weights for prediction. In next week’s exercise, you will write the backpropagation algorithm for learning the neural network parameters. 
# 
# We start by first reloading and visualizing the dataset which contains the MNIST handwritten digits (this is the same as we did in the first part of this exercise, we reload it here to ensure the variables have not been modified). 
#%%
#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

# get number of examples in dataset
m = y.size

# randomly permute examples, to be used for visualizing one 
# picture at a time
indices = np.random.permutation(m)

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

#%% [markdown]
# 
# ### 2.1 Model representation 
# 
# Our neural network is shown in the following figure.
# 
# ![Neural network](Figures/neuralnetwork.png)
# 
# It has 3 layers: an input layer, a hidden layer and an output layer.
# 
# You have been provided with a set of network parameters ($\Theta^{(1)}$, $\Theta^{(2)}$) already trained by us. These are stored in `ex3weights.mat`. The following cell loads those parameters into  `Theta1` and `Theta2`. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

#%%
# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary 
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)
#%% [markdown]
# <a id="section4"></a>
# ### 2.2 Feedforward Propagation and Prediction
# 
# Now you will implement feedforward propagation for the neural network. You will need to complete the code in the function `predict` to return the neural network’s prediction. You should implement the feedforward computation that computes $h_\theta(x^{(i)})$ for every example $i$ and returns the associated predictions. Similar to the one-vs-all classification strategy, the prediction from the neural network will be the label that has the largest output $\left( h_\theta(x) \right)_k$.
#%%
def feedfoward(X,Thetas):
    # Useful variables
    m = X.shape[0]

    # Column of 1's
    a = np.concatenate([np.ones((m, 1)), X], axis=1)

    for ii in range(len(Thetas)):
        # Layer weight 
        Theta = Thetas[ii]

        # Going through the layer 
        z = np.dot(a,Theta.T)
        a = utils.sigmoid(z)

        # Condition if already went through all layers
        if ii == len(Thetas)-1:
            return a
        else:
           a = np.concatenate([np.ones((m, 1)), a], axis=1) 
#%%
def predict(Theta1, Theta2, X):
    """  
    Instructions
    ------------
    Complete the following code to make predictions using your learned neural
    network. You should set p to a vector containing labels 
    between 0 to (num_labels-1).

    """
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    h = feedfoward(X,[Theta1, Theta2])
    p = np.argmax(h, axis=1)
    # =============================================================
    return p
#%% [markdown]
# Once you are done, call your predict function using the loaded set of parameters for `Theta1` and `Theta2`. You should see that the accuracy is about 97.5%.
#%%
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))
#%% [markdown]
# After that, we will display images from the training set one at a time, while at the same time printing out the predicted label for the displayed image. 
# 
# Run the following cell to display a single image the the neural network's prediction. You can run the cell multiple time to see predictions for different images.
#%%
if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')