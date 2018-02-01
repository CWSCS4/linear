import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D (each pixel), C (each class)) containing weights.
  - X: A numpy array of shape (N (each image), D (each pixel)) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :]
        dW[:, y[i]] -= X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW += 2 * W * reg
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Calculate scores
  scores = np.dot(X, W)

  print(scores.shape)

  # Calculate loss
  correct_scores = np.ones(scores.shape) * scores[y, np.arange(scores.shape[1])] # Correct scores
  theOne = np.ones(scores.shape)
  L = scores - correct_scores + theOne

  L[L < 0] = 0
  L[y, np.arange(scores.shape[1])] = 0 # Don't count y_i
  loss = np.sum(L)

  # Average over number of training examples
  num_train = X.shape[1]
  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad = np.zeros(scores.shape)

  L = scores - correct_scores + theOne

  L[L < 0] = 0
  L[L > 0] = 1
  L[y, np.arange(scores.shape[1])] = 0 # Don't count y_i
  L[y, np.arange(scores.shape[1])] = -1 * np.sum(L, axis=0)
  dW = np.dot(X.T, L)

  # Average over number of training examples
  num_train = X.shape[1]
  dW /= num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
