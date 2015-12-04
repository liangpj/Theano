import sys
import timeit
import os

import numpy
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams as Randomstreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA

class SdA(object) :
    """ Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking serveral
    dAs. The hidden layer of the dA at layer 'i' becomes that input of the
    dA at layer 'i+1'. The first layer dA gets as input the input of the SdA,
    and the hidden layer of the last dA represents the output. Note that after
    pretraining, the SdA is dealt with as a normal MLP, the dAs are only used to
    initialize the weights.
    """
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            n_ins=784,
            hidden_layers_sizes=[500,500],
            n_outs=10,
            corruption_levels=[0.1,0.1]
    ):
        """This class is made to support a variable number of layers

        :param numpy_rng: numpy random number generator used to draw initial weights

        :param theano_rng: Theano random generator; if None is given one is generated based
            on a seed drawn from 'rng'

        :param n_ins: dimension of the input to the sdA

        :param hidden_layers_sizes: intermediate layers size, must contain at least one value

        :param n_outs: dimension of the output of the network

        :param corruption_levels: amount of corruption to use for each layer
        """

        self.sigmoid_layer = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng :
            theano_rng = Randomstreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x') # the data is presented as rasterized images
        self.y = T.ivector('y') # the labels are presented as 1D vector of [int] labels

        # the Sda is an MLP, for which all wieghts of intermediate layers are shared with
        # a different denoising autoencoders we will first construct the SdA as  a deep
        # multilayer perceptron, and when constructing each sigmoidal layer we also construct
        # a denoising autoencoder that shares weights with that layer During pretraining we
        # will train these autoencoders (which will lead to chainging the weights of the MLP
        # as well) During finetuning we will finish training the SdA by doing stochastich gradient
        # descent on MLP
        # """

        for i in xrange(self.n_layers) :
            # construct the sigmoid layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer

            if i == 0 :
                input_size = n_ins
            else :
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first layer
            if i == 0:
                layer_input = self.x
            else :
                layer_input = self.sigmoid_layer[i-1].output

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layer.append(sigmoid_layer)
            # its arguably a philisophical question ...
            # but we are going to only declare that hte parameters of the
            # sigmoid_layers are parameters of the StackedDAA the visible biases
            # in the dA are parameters of those dA , but not the SdA
            self.params.extend(sigmoid_layer.params)

            # construct a denoising autoencoder that shared weights with this layer
            dA_layer = dA(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.W,
                bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # we now need to add a logistic layer on top of the MLP
        self.logLaer = LogisticRegression(
            input=self.sigmoid_layer[-1].output,
            n_in = hidden_layers_sizes[-1],
            n_out = n_outs)

        self.params.extend(self.logLaer.params)

        # construct a function that implements one step of finetunining

        # compute the cost for second phase of traning, defined as the
        # negative log likelihood
        self.finetune_cost = self.logLaer.negative_log_likehood(self.y)