import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename) as f:
        magic_num, num, rows, cols = struct.unpack('>4i', f.read(4*4))
        X = np.array(struct.unpack('>' + str(num*rows*cols) + 'B', f.read(num*rows*cols)), dtype=np.float32).reshape(num, -1) / 255.0

    with gzip.open(label_filename) as f:
        magic_num, num = struct.unpack('>2i', f.read(2*4))
        y = np.array(struct.unpack('>' + str(num) + 'B', f.read(num)), dtype=np.uint8)

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    part1 = ndl.log(ndl.summation(ndl.exp(Z), (-1, )))
    part2 = ndl.summation(Z * y_one_hot, (-1, ))
    return ndl.summation(part1 - part2) / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    assert len(y) % batch == 0, 'Batch_size is not right! There is remainder when len(y) // batch.'
    max_class_id = np.max(y)
    for i in range(len(y) // batch):
        # prepare data
        batch_X = X[i*batch:(i+1)*batch, :]
        batch_y = y[i*batch:(i+1)*batch]
        batch_one_hot_y = np.zeros((batch, max_class_id+1))
        batch_one_hot_y[np.arange(batch), batch_y] = 1

        # get data to tensor form
        X_tensor = ndl.Tensor(batch_X, requires_grad=False)
        y_tensor = ndl.Tensor(batch_one_hot_y, requires_grad=False)
        W1_tensor = ndl.Tensor(W1, requires_grad=True)
        W2_tensor = ndl.Tensor(W2, requires_grad=True)

        # forward and backward
        logits = ndl.matmul(ndl.relu(ndl.matmul(X_tensor, W1_tensor)), W2_tensor)
        loss = softmax_loss(logits, y_tensor)
        loss.backward()

        # update parameters
        W1 += - lr * W1_tensor.grad.numpy()
        W2 += - lr * W2_tensor.grad.numpy()

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
