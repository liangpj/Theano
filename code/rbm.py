import timeit
import os

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data


class RBM(object) :
    """Restricted Boltzmann Machine(RMB)"""
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
         RBM constructor. Defines the parameters of hte model along with
         basic operations for inferring hidden from visible (and vice-versal),
         as well as for performing CD updates.

        :param input: None for standalone RMBs or symbolic variable if RMB is part
        of a larger graph

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RMBs or symbolic variable pointing to a shared weight
        matrix in case RBM is part of a DBN network; in a DBN, the wegihts are shared between
        RMBs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing to a shared hidden units
        bias vector in case RMB is part of a different network

        :param vbias: None for standalone RBMs or a symbolic variable pointing to a shared visible units bias
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None :
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with 'initial_W' which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(5./(n_hidden + n_visible)) the ouput of uniform if
            # converted using asarray to dtype theaon.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6./(n_hidden + n_visible)),
                    high= 4 * numpy.sqrt(6./ (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variable for weights and bias
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None :
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(n_hidden,
                                  dtype=theano.config.floatX),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared varible for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(n_visible,
                                  dtype=theano.config.floatX),
                name='vbias',
                borrow=True)

        # initialize input layer for standalon RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        # **** WARNING: it is not a good iead to put things in this list
        # other than shared variables created in this function
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, vis):
        """
        this function propagates the visible units activation upwards to
        the hidden units

        Note that we return als the pre-sigmoid activation of the layer.
        As it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be nedded to write down a more stable coputational
        graph(see details in hte recontruction cost function)
        """
        ## compute p(h|x) = sigmoid(W * x + hbias)

        pre_sigmoid_activatoin = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activatoin, T.nnet.sigmoid(pre_sigmoid_activatoin)]

    def sample_h_given_v(self, v0_samle):
        """
        This function infers state of hidden units given visible units
        """
        # compute the activation of the hidden units given a sample of the visible
        pre_sigmoid_h1, h1_mean = self.propup(v0_samle)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial return a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to reuturn the dypte floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1,
                                             p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """
        this function propagates the hidden units activation downwards to the
        visible units

        Note that we return als the pre_sigmoid_activation of the layaer. As it
        will turn out later, due to how theano deals with optimizations, this symbolic
        variable will be needed to write down a more stable computational graph(see details
        in the reconstruction cost function)
        """

        # compute the p(x|h) = sigmoid(W.T * h + b)
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """
        this function infers state of visible units given hidden units
        """
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        v1_sample = self.theano_rng.binomial(size= v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """
        This function implements one step of Gibbs sampling,
        sartting from the hidden state
        """

        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """
        This function implements one step of Gibbs sampling,
        starting from the visible state
        """

        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)

        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        """
        Function to copute the free energy
        """

        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)

        return  -vbias_term - hidden_term


    def get_cost_update(self, lr=0.1, persistent=None, k=1):
        """
        This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable containing
        old state of Gibbs chain. This must be a shared variable of size (batch size,
        number of hidden units.)

        :param k: number of Gibbbs steps to do in CD-k/PCD-k

        :return:  Returns a proxy for the cost and the updates dictionary. The dicitonary conatins
        the update rules for weights and biases but also an update of the shared variable used to
        store the persistent chain, if one is used
        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None :
            chain_start = ph_sample
        else :
            chain_start = persistent

        # perform actual negative phase in order to implement CD-k/PCD-k we need
        # to scan over the function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information:
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # determinate gradients on RBM parameters note that we only
        # need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # we must no compute the gradient through the gibbs sampling
        gparams = T.grad(cost=cost, wrt=self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params) :
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
            if persistent:
                # Note that this works only if persistent is a shared variable
                updates[persistent] = nh_samples[-1]
                # pseudo-likelihood is a better proxy for CD
                monitoring_cost = self.get_pseudo_likelihood_cost(updates)
            else :
                # reconstructing cross-entropy is a better proxy for CD
                monitoring_cost = self.get_reconstruction_cost(updates,
                                                               pre_sigmoid_nvs[-1])
        return monitoring_cost, updates


    def get_pseudo_likelihood_cost(self, updates):
        """
        Stochastic approximation to the pseudo-likelihood
        """

        # index of bit i in expression p(x_i | x_{\i} )
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bi_x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[;bit_i_idx] = 1 - xi[;,bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        #calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))
        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """ Approximation to the recontruction error

        Note that this function requires the pre-sigmoid actiavtion as input. To
        undertstand why this is so you need to understand a bit about how Theano works.
        Whenever you compile a Theano function, the computational graph that you pass as input
        gets optimized for speed and stability. This is done by changing several parts of
        the subgraphs with others. One such optimization expresses terms of softplus. We need this
        optimizations for the cross-entropy since sigmoid of numbers larger than 30. (or even less
        then that) return to 1. and numbers of smaller than -30, turn to 0 which ini terms will force
        theano to compute log(0) and thereforce we will get either -inf or NaN as cost. If the value is
        expressed in terms of softplus we do not get this undersirable behaviour. This optimiation usually
        works fine, but here we have a special case. The sigmoid is applied inside the scan op, while
        the log is outisde. Therefore Theano will only see log(scan(...)) instead of log(sigmoid(..))
        and will not apply the wanted optimization. We can not go and replace the sigmoid in scan
        with something else alse, because this only needs to be done on the last step. Therefore the
        easiest adn more efficient way is to get also teh pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan sunch that Theano can catch and optimize
        the expression.
        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                ( 1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return  cross_entropy

def test_rbm(leanrning_rate = 0.1, training_epochs = 15,
             dataset='data/mnist.pkl.gz', batch_size = 20,
             n_chains = 20, n_samples=10, output_folder='rmb_res',
             n_hidden=500):
    """
    Demonstrate how to trian and afterwards sample from it using Theano.

    This is demonstrated on MNIST

    :param leanrning_rate: learnning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the pickled dadtdaset

    :param batch_size: size of a batch used to trian the RBM

    :param n_chains: number of parallel Gibbs chains to used for sampling

    :param n_samples:  number samples to plot for each chain
    """
    datasets = load_data(dataset)

    train_set_x, trian_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # compute nunmber of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 * 30))

    # initialize storage for the persistent chain (state - hidden
    # layer of cahin)
    persistent_chain = theano.shared(
        value=numpy.zeros((batch_size, n_hidden),
                          dtype=theano.config.floatX),
                          borrow = True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible= 28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_update(lr=leanrning_rate, persistent=persistent_chain,
                                        k=15)

    #######################
    ## TRAINING THE TBM  ##
    #######################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of trian_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x : train_set_x[index * batch_size : (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs) :
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches) :
            mean_cost += [train_rbm(batch_index)]

        print "Training epoch %d, cost is " % epoch, numpy.mean(mean_cost)

        # plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X = rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10,10),
                tile_spacing=(1,1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print('Training took %f minutes' % (pretraining_time / 60.))

    ###########################
    ## Sampling from the RBM ##
    ###########################

    # find out the nubmer of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples,with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                      dtype=theano.config.floatX)
    )

    plot_every = 1000
    # define one step of Gibbs sampling(mf = mean-filed) define a function
    # that does 'plot_every' steps before returning the smaple for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(rbm.gibbs_vhv,
                    outputs_info=[None, None, None, None, None, persistent_vis_chain],
                    n_steps=plot_every)

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that impleemnts our persistent chain.
    # we generate the 'mean field' activations for plotting and the
    # actual samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([],
                                [vis_mfs[-1],
                                 vis_samples[-1]],
                                updates=updates,
                                name='sample_fn')

    # create a space t store the image for plotting( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains -1),
        dtype='uint8'
    )
    for idx in xrange(n_samples) :
        # generate plot_every intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print '... plotting sample ', idx
        image_data[29 * idx: 29 * idx + 28, :] = tile_raster_images(
            X = vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1,1)
        )

    # construct image
        image = Image.fromarray(image_data)
        image.save('samples.png')
        os.chdir('../')

if __name__ == '__main__' :
    test_rbm()
