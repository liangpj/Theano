#coding:utf-8

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object) :
    """
    Pool Layer of a convolutional network 卷积神经网络
    """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :param rng: a random number generator used to initialize weights

        :param input: symbolic image tensor, of shape image_shape

        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)

        :param image_shape: (batch size, num input feature maps, image height, image width)

        :param poolsize: the downsampling (pooling) factor(#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are 'num input feature maps * filter height * filter witdth'
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # 'num output feature map * filter height * filter width' / poolingsize
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_boud = numpy.sqrt(6./(fan_in + fan_out))
        self.W = theano.shared(
            value=numpy.asarray(
                rng.uniform(low=-W_boud, high=W_boud, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolue input feature maps with filters
        conv_out = conv.conv2d(
            input = input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape(1, n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='data/mnist.pkl.gz', nkearns=[20, 50], batch_size=500):
    """
    Demonstrates lenet on MNIST dataset

    :param learning_rate: learnning rate used (factor for the stochastic gradient)

    :param n_epochs: maximal number of epochs to run the optimizer

    :param dataset: path to the dataset used for training /testing (MNIST here)

    :param nkearns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch

    x = T.matrix('x') # the data is presented as rasterized images
    y = T.ivector('y') # the labels are presented as 1D vector of [int] labels

    #######################
    ## BUILD ACTUAL MODEL #
    #######################
    print '... building the model '

    # Reshape matrix of rasterized image of shape(batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28,28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to(28-5+1, 28-5+1) =(24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape(batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(rng,
                                input=layer0_input,
                                image_shape=(batch_size, 1, 28, 28),
                                filter_shape=(nkearns[0], 1, 5, 5),
                                poolsize=(2,2)
                                )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12 - 5 + 1, 12 - 5 + 1) = (8,8)
    # maxpooling reduces this further to(8/2, 8/2) = (4,4)
    # 4D output tensor is thus of shape( batch_size, nkers[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkearns[0], 12, 12),
        filter_shape=(nkearns[1], nkearns[0], 5, 5),
        poolsize=(2,2)
    )

    # the HiddenLayer beging fulling-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # this will generate a matrix of shape(batch_size, nkerns[1]*4*4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkearns[1] * 4 * 4,
        n_out = 500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likehood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size : (index + 1) * batch_size],
            y: test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradient for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGG Since this model has many parameters. it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    #################
    ## TRAIN MODEL ##
    #################
    print '... training'
    # early-stopping parameters
    patience = 10000 # look as this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found

    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience/2)
                                # go throught this many
                                # minibache before checking the network
                                # on the validation set; in this case we
                                # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping) :
        epoch += 1
        for minibatch_index in xrange(n_train_batches) :
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if ( iter + 1) % validation_frequency == 0 :

                # compute zero-one loss on validation set
                validation_loss = [validate_model(i) for i
                                   in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_loss)
                print ('epoch %i, minbatch %i / %i, validation error %f %%' %
                       (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss :

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * this_validation_loss:

                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print ('            epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%' %
                           (epoch, minibatch_index + 1, n_test_batches,
                            test_score * 100.))
                if patience <= iter :
                    done_looping = True
                    break

    end_time = timeit.default_timer()
    print('Optimization complete. ')
    print ('Best validation score of %f %% obtained at iteration %i, '
           'with best performance %f %%' %
           (best_validation_loss*100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file' +
                          os.path.split(__file__)[0] +
                          'ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__' :
    evaluate_lenet5()