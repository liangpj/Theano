#coding:utf-8

import sys
import timeit
import os

import theano
import theano.tensor as T
import numpy

from logistic_sgd import LogisticRegression, load_data

###
# reference: http://deeplearning.net/tutorial/mlp.html#tips-and-tricks-for-training-mlps
###

""" 只有一个隐藏层的神经网络，其默认情况下激函数为tanh
    输出层使用logistic sigmoid 函数进行分类
"""

class Hidden(object) :
    """定义一个隐藏层的类。
    """

    def __init__(self, rng, input, n_in,n_out, W=None, b=None,
                 activation=T.tanh):
        """初始化隐藏层的参数

        :param rng: 随机数产生器
        :type rng: numpy.random.RandomState

        :param input: 输入数据矩阵
        :type input: T.dmatrix

        :param n_in: 输入数据集features数
        :type  int

        :param n_out: 隐藏层神经元的个数
        :type int

        :param W: 权重
        :param b: 偏差

        :param activation: 激函数
        """
        self.input = input

        # 权值初始化
        # 判断是否提供初始化值
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(low=-numpy.sqrt(6./(n_in + n_out)),
                            high=numpy.sqrt(6./(n_in + n_out))
                            ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid :
                W_values *= 4   # 注意
            W = theano.shared(
                value=W_values,
                name='W',
                borrow=True
            )
        if b is None:
            b = theano.shared(
                value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        lin_out = T.dot(input, self.W) + b
        self.output = (
            lin_out if activation is None
            else activation(lin_out)
        )

        # 隐藏层参数
        self.params = [self.W, self.b]

class MLP(object) :
    """
    Multi-Layer Perceptron Class
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """

        :param rng: numpy.random.RandState

        :param input: T.tensorType

        :param n_in: 输入层feature数

        :param n_hidden: 隐藏层单元数

        :param n_out: 输出层labels数
        """
        # 定义隐藏层
        self.hiddenLayer = Hidden(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # 定义输入层
        # 使用Multi Class Logistic Regression
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # Regularization
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # 定义 negative log likelihood function
        self.negative_log_liklihood = (
            self.logRegressionLayer.negative_log_likehood
        )

        # 定义误差函数
        self.errors = self.logRegressionLayer.errors

        # 模型参数
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0., L2_reg=0.0001, n_epochs=1000,
             dataset='data/mnist.pkl.gz', batch_size=20, h_hidden=500) :

    ##################
    ## LOAD DATASET ##
    ##################

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算 随机梯度下降迭代次数
    n_train_batches = train_set_x.get_value().shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] / batch_size
    n_test_batches = test_set_x.get_value().shape[0] / batch_size

    #allocate the symbolic variables for the data
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar('index')

    rng = numpy.random.RandomState(1234)

    #创建MLP类
    classifier = MLP(
        rng=rng,
        input=[index],
        n_in=28*28,
        n_hidden=h_hidden,
        n_out=10
    )

    # cost function
    cost = (
        classifier.negative_log_liklihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # 对参数进行求导
    gparams = [
        T.grad(cost=cost, wrt=param) for param in classifier.params
    ]

    # 随机梯度下降跟新公式
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # 构建训练模型
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # 构建validation Model
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size : (index + 1) * batch_size],
            y: valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # 构建测试模型
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size : (index + 1) * batch_size],
            y: test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    #################
    ## TRAIN MODEL ##
    #################
    print '... train model'

    #设置early-stopping 参数
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping) :
        epoch += 1
        for minibatch_index in xrange(n_train_batches) :
            minbatch_avg_cost = train_model(minibatch_index)

            iter = (epoch -1) * n_train_batches + minibatch_index
            if ( iter + 1) % validation_frequency == 0 :
                validation_losses = [validate_model[i] for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print (
                    " epoch %i, minibatch %i / %i, validation error %f %%"
                    %(
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss :
                    if (
                        this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        #对该模型在测试集上进行数据测试
                        test_losses = [ test_model(i)
                                        for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                    print (
                        "      epoch %i, minibatch %i / %i, test error of "
                        %(
                            epoch,
                            minibatch_index + 1,
                            n_test_batches,
                            test_score * 100.
                           ))
            if iter >= patience :
                done_looping = True
                break

    end_time = timeit.default_timer()

    print (
        "Optimization complete. Best validation socre of %f %%"
        'obatained at iteration %i, with test performance %f %%'
        %(
            this_validation_loss * 100,
            best_iter + 1,
            test_score * 100.
        )
    )

    print >> sys.stderr, ("The code for file "
                          + os.path.split(__file__)[1] +
                          'ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__' :
    test_mlp()