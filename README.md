
Module 3: Logistic Regression
In this assignment, you will

explore the sigmoid function (also known as the logistic function)
explore logistic regression; which uses the sigmoid function
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_data
Sigmoid or Logistic Function
As discussed in the lecture videos, for a classification task, we can start by using our linear regression model, 
, to predict 
 given 
.

However, we would like the predictions of our classification model to be between 0 and 1 since our output variable 
 is either 0 or 1.
This can be accomplished by using a "sigmoid function" which maps all input values to values between 0 and 1.
Let's implement the sigmoid function and see this for ourselves.

Formula for Sigmoid function
The formula for a sigmoid function is as follows -

 

In the case of logistic regression, z (the input to the sigmoid function), is the output of a linear regression model.

In the case of a single example, 
 is scalar.
in the case of multiple examples, 
 may be a vector consisting of 
 values, one for each example.
The implementation of the sigmoid function should cover both of these potential input formats. Let's implement this in Python.
NumPy has a function called exp(), which offers a convenient way to calculate the exponential ( 
) of all elements in the input array (z).

It also works with a single number as an input, as shown below.

# Input is an array.
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)
Input to exp: [1 2 3]
Output of exp: [ 2.71828183  7.3890561  20.08553692]
Input to exp: 1
Output of exp: 2.718281828459045
The sigmoid function is implemented in python as shown in the cell below.

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    #####################################
    # WRITE THE SIGMOID FUNCTION HERE
    g = 1 / (1 + np.exp(-z))

    #####################################

    return g
Let's see what the output of this function is for various value of z

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3)
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])
Input (z), Output (sigmoid(z))
[[-1.000e+01  4.540e-05]
 [-9.000e+00  1.234e-04]
 [-8.000e+00  3.354e-04]
 [-7.000e+00  9.111e-04]
 [-6.000e+00  2.473e-03]
 [-5.000e+00  6.693e-03]
 [-4.000e+00  1.799e-02]
 [-3.000e+00  4.743e-02]
 [-2.000e+00  1.192e-01]
 [-1.000e+00  2.689e-01]
 [ 0.000e+00  5.000e-01]
 [ 1.000e+00  7.311e-01]
 [ 2.000e+00  8.808e-01]
 [ 3.000e+00  9.526e-01]
 [ 4.000e+00  9.820e-01]
 [ 5.000e+00  9.933e-01]
 [ 6.000e+00  9.975e-01]
 [ 7.000e+00  9.991e-01]
 [ 8.000e+00  9.997e-01]
 [ 9.000e+00  9.999e-01]
 [ 1.000e+01  1.000e+00]]
The values in the left column are z, and the values in the right column are sigmoid(z). As you can see, the input values to the sigmoid range from -10 to 10, and the output values range from 0 to 1.

Now, let's try to plot this function using the matplotlib library.

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
Text(0.5, 0, 'z')

As you can see, the sigmoid function approaches 0 as z goes to large negative values and approaches 1 as z goes to large positive values.

Decision Boundary
Lets the decision boundary for a logistic regression model. This will give you a better sense of what the model is predicting.

Let's suppose you have following training dataset

The input variable X is a numpy array which has 6 training examples, each with two features
The output variable y is also a numpy array with 6 examples, and y is either 0 or 1
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
Logistic regression model
Suppose you'd like to train a logistic regression model on this data which has the form


where 
 
, which is the sigmoid function

Let's say that you trained the model and get the parameters as 
. That is,


(You'll learn how to train/fit and get these parameters in the further sections)

Let's try to understand what this trained model is predicting by plotting its decision boundary

Recall that for logistic regression, the model is represented as

 

where 
 is known as the sigmoid function and it maps all input values to values between 0 and 1:

 
 and 
 is the vector dot product:


We interpret the output of the model (
) as the probability that 
 given 
 and parameterized by 
 and 
.

Therefore, to get a final prediction (
 or 
) from the logistic regression model, we can use the following heuristic -

if 
, predict 

if 
, predict 

Let's plot the sigmoid function to see where 

Plotting decision boundary
Now, let's go back to our example to understand how the logistic regression model is making predictions.

Our logistic regression model has the form


From what you've learnt above, you can see that this model predicts 
 if 

Let's see what this looks like graphically. We'll start by plotting 
, which is equivalent to 
.

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data

plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

Train a logistic regression model using scikit-learn
https://scikit-learn.org/0.16/modules/generated/sklearn.linear_model.LogisticRegression.html

Create a Regression Object
Call fit function
Get predictions
Get Score of model
Get coefficients (w0, w1) and intercept(b)
from sklearn.linear_model import LogisticRegression
# Create a Regression Object
model = LogisticRegression()

# Call fit function
model.fit(X, y.ravel())

# Get predictions
preds = model.predict(X)

# Get Score of model
score = model.score(X, y)

# Get coefficients (w0, w1) and intercept(b)
w = model.coef_
b = model.intercept_

print("Predictions:\n", preds)
print("Score:", score)
print("Weights:", w)
print("Intercept:", b)
Predictions:
 [0 0 0 1 1 1]
Score: 1.0
Weights: [[0.904 0.736]]
Intercept: [-2.334]
