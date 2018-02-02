import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
      # margin = S_k - S_y_i + 1 
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # if margin > 0, then S_k < S_y_i - 1

      # dL/dW = 0 if S_k < S_y_i - 1 --> by default

      # S_k - S_y_i + 1 > 0 --> S_k > S_y_i - 1
      if margin > 0:
        loss += margin

        # dL/dW = X[i] if S_k > S_y_i - 1
        dW[:, j] += X[i] # need to add, so it becomes the sum

        # dL/dW --> need to account for -X[i] in W[j]*X[i] - W[y[i]] * x[i] --> as dL/dW[y[i] = -x[i], as gradient is derivative w.r.t all weights.
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Average the gradients, just like the loss
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # add the bias to (regularize) the gradient
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # I had to use http://cs231n.github.io/optimization-1/#gradcompute in addition to my notes to do this assignment.
  # code is integrated above

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

  # S is scores matrix (N, C)
  S = np.dot(X, W)

  # An N-dimensional vector with the score that the correct class for each
  # image receives
  correct_class_scores = S[np.arange(X.shape[0]), y]

  # N by C matrix
  score_margins = np.maximum(0, S - correct_class_scores.reshape([-1,1]) + 1)
  score_margins[np.arange(X.shape[0]),y] = 0
  loss = np.sum(score_margins) / X.shape[0] + reg * np.sum(W*W)


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
  
  # gradient calculations

  # (C, N) array, so that when multiplied by (N, D) data, you get
  # your gradient.
  pixel_coeffs = (score_margins.T > 0).astype(np.float64)
  negative_coeffs = np.sum(pixel_coeffs, axis = 0)
  pixel_coeffs[y, np.arange(X.shape[0])] -= negative_coeffs
  dW = pixel_coeffs.dot(X).T
  dW /= X.shape[0]
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
