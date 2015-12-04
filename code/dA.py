import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

try :
    import PIL.Image as  Image
except ImportError :
    import Image

from logistic_sgd import load_data
from utils import tile_raster_images

class dA(object) :

    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units(the dimension
        d of the input), the number of hidden units(the dimension d'of the latent or hidden
        space) and the corruption level. The constructor also receives symbolic variables for
        the input,weights and bias. Such a symbolic variables are useful when, for example the
        input is the result of some computions, or when weights are shared between the dA and
        an MLP layer. When dealing with SdAs this always happens, the dA on layer 2 gets as input
        the output of the dA on layer 1, and the weights of the dA are used in the second stage
        of training to contruct an MLP

        :param numpy_rng: number random genrator used to generate weights

        :param theano_rng: Thenao random genrator; if None is given one is generated based on
        seed drawn from 'rng'

        :param input:  a symbolic description of the input or None for standalone dA

        :param n_visible: number of visible units

        :param n_hidden:  number of hidden units

        :param W: Theano variable pointing to a set of weights that should be shared belong the
            dA and another architecture; if dA should be standalone set this to None

        :param bhid:  theano variable pointing to a set of biases values (for hidden units) that
            should be shared belong dA and another architecture; if dA should be standalone set
            this to None

        :param bvis: theano variable pointing to a set of biases values (for visible units) that
            should be shared belong dA and another architecture; if dA should be standalone set
            this to None

        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create Theano random generator that givens symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note W' was written as 'W_prime' and b' as 'b_prime'
        if not W :
            # W is initialized with 'initial_W' which is uniformely sampled from
            # -4*sqrt(6./(n_visible + n_hidden) and 4*sqrt(6./(n_visible + n_hidden)
            # the output of uniform of converted using asarray to dtype theano.config.floatX
            # so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(low=-4*numpy.sqrt(6./(n_visible + n_hidden)),
                                  high=4*numpy.sqrt(6./(n_visible + n_hidden)),
                                  size=(n_visible, n_hidden)),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(n_visible, dtype=theano.config.floatX),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(n_hidden, dtype=theano.config.floatX),
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime correspond to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of serveral
            # examples, each example being row
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps '1 - corruption_level' entries of the inputs
        the same and zero-out randomly selected subset of size 'corruption_level
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p = 1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_recontructed_input(self, hidden):
        """
        Computes the recontructed input given the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        This function computes the cost and the updates for one training steop of the dA
        """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_recontructed_input(y)
        # Note : we sum over the size of a datapoint; if we are using
        # minibataches. L will be a vector, with one entry per example
        # in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each elements is the cross-entropy cost
        # of the recontruction of the corresponding example of the minibatch. We
        # need to compute the average of all these to get the cost of the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the 'dA' with respect to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)  for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def test_dA(learning_rate = 0.1, training_epochs=15, dataset='data/mnist.pkl.gz',
            batch_size=20, output_foldr='dA_res') :

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute the number of minibatches for training, validataion and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar('index') # index to a [mini]batch
    x = T.matrix('x') # the data is presented as rasterized images

    if not os.path.isdir(output_foldr) :
        os.makedirs(output_foldr)
    os.chdir(output_foldr)


    #####################################
    # BULIDING THE MODEL NO CORRUPTION ##
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(corruption_level=0, learning_rate=learning_rate)

    train_da = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1)* batch_size]
        }
    )

    start_time = timeit.default_timer()
    ##############
    ## TRAINING ##
    ##############

    # go through training epochs
    for epoch in xrange(training_epochs) :
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches) :
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = end_time - start_time

    print >> sys.stderr, ('The no corruption code for file' +
                          os.path.split(__file__)[1] +
                          'ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10,10),
                           tile_spacing=(1,1)))
    image.save('fileters_corruption_0.png')

    #######################################
    ## BUILDING THE MODEL CORRUPTION 30% ##
    #######################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(corruption_level=0.3, learning_rate=learning_rate)

    train_da = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
         x : train_set_x[index*batch_size : (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()
    ##############
    ## TRAINING ##
    ##############

    # go through training epochs
    for epoch in xrange(training_epochs) :
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches) :
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()
    training_time = end_time - start_time

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2f ' % ((training_time) / 60.))

    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10,10),
        tile_spacing=(1,1)))
    image.save('filters_corruption_30.png')

    os.chdir('../')

if __name__ == '__main__' :
    test_dA()