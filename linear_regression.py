'''
Take data -> start with random Wx + b -> calculate loss -> adjust W and b using gradient descent 
'''
from typing import Union # allows for type hints in class parameters
from random import random
# from time import sleep
import numpy as np
from numpy.typing import ArrayLike as ArrayLike # for typing
from scipy.sparse import spmatrix
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression: # use scikit-learn documentation as reference structure
    def __init__(self, tol = 1e-6, learning_rate = 0.01): 
        self.tol = tol # tolerance will be convergence critera for grad descent
        self.learning_rate = learning_rate
        self.fitted = False

    def fit(self, x: Union[np.ndarray, list, pd.DataFrame, spmatrix], y: Union[np.ndarray, list, pd.DataFrame, spmatrix]):
        # x should have format: [[x1,x2...], [x1,x2,...]] (sub arrays are different observations)
        # y has format: [y1,y2...] (each y is the actual value per observation)
        '''
        Fits the linear regression model to the provided data.
        Args:
            x (Union[np.ndarray, list, pd.DataFrame, spmatrix]): Input features.
            y (Union[np.ndarray, list, pd.DataFrame, spmatrix]): Target values.
        Raises:
            ValueError: If x is not a 2D array or if rows in x do not have the same length.
            RuntimeError: If the model is already fitted.
        '''
        self.x = np.array(x) # ensures class variables are internalized as numpy arrays
        if self.x.ndim == 1:
            raise ValueError("x should be a 2D array, not a 1D array.")
        # Example: x is a list of lists
        if isinstance(x, list):
            row_lengths = [len(row) for row in x]
            if len(set(row_lengths)) != 1:
                raise ValueError("All rows in x must have the same length.")
        self.y = np.array(y) 
        self.W = np.random.randn(self.x.shape[1])
        self.b = random()
        self.grad_descent(self.x,self.y,self.W,self.b)
        self.fitted = True
    
    def calc_grad(self, x: np.ndarray, y: np.ndarray, W: np.ndarray, b: float):
        # multivariate MSE has form: 1/2n * ||X*theta - y||^2, where theta = [W, b], n is the number of samples 
        # Note: given matrix A: A^T * A
        '''
        Calculates the cost function for linear regression.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
            W (np.ndarray): Weights.
            b (float): Bias term.
        '''
        n = x.shape[0]
        predictions = np.dot(x, W) + b # calculate dot product of x and W, then add bias to all terms
        errors = predictions - y
        # cost = (1 / (2 * n)) * np.sum(errors ** 2) # gets aggregate cost 
        # gradient descent update rule: θnew = θ - α*∇J(θ)
        # θ - vector of parameters [W,b]
        # α - hyperparameter, learning rate
        # J(θ) - cost function given these parameters
        # ∇J(θ) - gradient of cost function

        # ∇wJ(θ) = ∇wJ(W, b) = (∂ / ∂W) * (1/2n * ||X*theta - y||^2)
        # by the identity: ||A||^2 = A^T * A. We get (∂ / ∂W) * ((1/2n * ||X*theta - y||^2)^T * (1/2n * ||X*theta - y||^2))
        # Therefore ∇wJ(θ) = 1/2n * 2 * (1/2n * ||X*theta - y||^2)
        dw = (1 / n) * np.dot(x.T, errors)
        
        # ∇bJ(θ) = ∇bJ(W, b) = (∂ / ∂b) * (1/2n) * ||X*W + b - y||^2)
        # ∇bJ(θ) = (1 / 2n) * 2 * error
        db = (1 / n) * np.sum(errors)
        grad =  np.concatenate((dw, np.array([db]))) # concatenate the gradients for W and b
        return dw, db, grad

    def grad_descent(self, x: np.ndarray, y: np.ndarray, W: np.ndarray, b: float):
        # ∇wJ(θ) = ∇wJ(W, b) = (∂ / ∂W) * (1/2n * ||X*theta - y||^2)
        # by the identity: ||A||^2 = A^T * A. We get (∂ / ∂W) * ((1/2n * ||X*theta - y||^2)^T * (1/2n * ||X*theta - y||^2))
        # Therefore ∇wJ(θ) = 1/2n * 2X^T * (1/2n * ||X*theta - y||^2)
        '''
        Performs gradient descent to optimize the weights and bias.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
            W (np.ndarray): Initial weights.
            b (float): Initial bias term.
        '''
        
        while True: 
            dw, db, grad = self.calc_grad(self.x, self.y, self.W, self.b)
            if np.linalg.norm(grad) < self.tol:
                break
            print("grads", dw, db, grad)
            print(dw, db)
            # sleep(0.05)
            self.W -= self.learning_rate*dw
            self.b -= self.learning_rate*db
        print(self.W)
    
    def score(self, x=None, y=None):
        '''
        Computes the R^2 score of the model. Defined as 1 - (u / v), 
        where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and
        v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Returns:
            float: R^2 score of the model.
        Raises:
            RuntimeError: If the model is not fitted.
        '''
        if not self.fitted:
            raise RuntimeError("Fit the model first")
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        
        y_pred = np.dot(x, self.W) + self.b
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        print(v)
        if v == 0:
            # If y is constant and predictions are perfect, return 1.0; else 0.0
            return 1.0 if 1 - 1 - u / v < self.tol else 0.0
        return 1.0 if 1 - 1 - u / v < self.tol else 1 - u / v
    
    def predict(self, x: Union[np.ndarray, list, pd.DataFrame, spmatrix]):
        '''
        Predicts target values for the given input features.
        Args:
            x (Union[np.ndarray, list, pd.DataFrame, spmatrix]): Input features.
        Returns:
            np.ndarray: Predicted target values.
        Raises:
            RuntimeError: If the model is not fitted.
        '''
        if not self.fitted:
            raise RuntimeError("Fit the model first")
        x = np.array(x)
        if self.W.ndim == 1:
            return x @ self.W + self.b
        else:
           return x @ self.W.T + self.b
        
    def plot_regression2d(self):
        '''
        Plots the data points (x, y) and the regression line defined by y = W0x0 + W1x1 + ... + Wnxn + b.
        '''
        plt.style.use('default')
        plt.style.use('ggplot')

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.x, self.predict(self.x), color='k', label='Regression model')
        ax.scatter(self.x, self.y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
        ylabel = input("Enter the y-axis label: ")
        ax.set_ylabel(ylabel, fontsize=14)
        xlabel = input("Enter the x-axis label: ")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.legend(facecolor='white', fontsize=11)
        ax.set_title('$R^2= %.2f$' % self.score(), fontsize=18)

        fig.tight_layout()
        plt.show()
    
    def plot_regression3d(self):
        """
        Plots a 3D scatter plot of the data and the regression surface.
        """
        # Prepare data for plotting
        x = self.x[:, 0]
        y = self.x[:, 1]
        z = self.y

        # Create grid for the regression surface
        x_pred = np.linspace(x.min(), x.max(), 30)
        y_pred = np.linspace(y.min(), y.max(), 30)
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        model_viz = np.column_stack([xx_pred.ravel(), yy_pred.ravel()])
        z_pred = self.predict(model_viz).reshape(xx_pred.shape)

        # Set up figure and axes
        fig = plt.figure(figsize=(12, 4))
        views = [(28, 120), (4, 114), (60, 165)]
        axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

        # Labels for the axes
        x_label = input("Enter the x-axis label: ")
        y_label = input("Enter the y-axis label: ")
        z_label = input("Enter the z-axis label: ")

        for ax, view in zip(axes, views):
        # Scatter plot of actual data
            ax.scatter(x, y, z, color='k', marker='o', alpha=0.5, label='Data')
            # Surface plot of regression
            ax.plot_surface(xx_pred, yy_pred, z_pred, color='#70b3f0', alpha=0.4, edgecolor='none')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.view_init(elev=view[0], azim=view[1])
    
        plt.tight_layout(w_pad=0.3)
        plt.show()