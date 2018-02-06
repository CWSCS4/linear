import numpy as np
from random import shuffle

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
	scores = np.dot(X, W)
	num_classes = W.shape[1]
	num_train = X.shape[0]

	for i in range(num_train):

		correct_value = y[i]
		score = scores[i]

		# calculate probability for a certain class, add the correct one to the loss
		prob = np.exp(score) / np.sum(np.exp(score))
		loss += np.log(prob[correct_value])

		# preemptively handle gradient subtracting
		prob[y[i]] -= 1

		# calculate gradient by 
		for j in range(num_classes):
			#prob = np.exp(scores[j]) / np.sum(np.exp(scores))

			dW[:, j] += prob[j] * X[i, :]


	# calculate loss - negate because has been adding + regularize
	loss = -loss / len(X) + np.sum(W ** 2) * reg * 0.5

	# regularize gradient
	dW /= num_train
	dW += reg * W

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
	num_train = X.shape[0]

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	
	scores = np.dot(X, W) # get the scores
	# find the probabilities of each class, keeping it as an array (keepdims=True)
	probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) 
	probs = probs.T

	# find the probability for the correct classes
	correct_probs = probs[y, np.arange(num_train)]
	# calculate softmax loss
	loss = np.sum(-np.log(correct_probs))

	# regularize the loss
	loss /= num_train
	loss += np.sum(W ** 2) * reg * 0.5

	# calculate the gradient - first, subtract 1 from correctly classified
	probs[y, np.arange(num_train)] -= 1

	# calculate the dot product, regularize
	dW = np.dot(probs, X) / num_train
	dW = dW.T
	dW += reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

