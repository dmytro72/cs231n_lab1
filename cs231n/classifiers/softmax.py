import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i, sample in enumerate(X):
    e_sum = 0
    exps = []
    for j, param in enumerate(W.T):
      f = sample.dot(param)
      exp = np.exp(f)
      exps.append(exp)
      e_sum += exp
      if y[i] == j:
        e = exp
    loss += -np.log(e / e_sum)

    # calculate derivatives
    for j, exp in enumerate(exps):
      dW[:, j] += exp / e_sum * sample
      if y[i] == j:
        dW[:, j] -= sample
  loss /= y.size
  dW /= y.size
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # normalization Wx matrix
  scores -= np.max(scores, axis=1, keepdims=True)
  exp = np.exp(scores)
  exp_sum = np.sum(exp, axis=1, keepdims=True)
  loss = np.sum(-np.log(np.divide(exp[np.arange(y.size), y][:, np.newaxis] ,exp_sum)))
  loss /= y.size
  loss += reg * np.sum(W*W)

  x_entities = np.zeros_like(scores)
  x_entities[np.arange(y.size), y] = 1
  dW = X.T.dot(np.divide(exp ,exp_sum) - x_entities)
  dW /= y.size
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

