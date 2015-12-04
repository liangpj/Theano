#coding:utf-8

import cPickle
import os
import timeit
import sys
import gzip

import theano
import theano.tensor as T
import numpy



""" 多类logistic 回归
"""

class LositicRegression(object) :
    """
    # 初始化Multi-class Logistic Regression 参数
    # y = softmax(x * W + b)
    """

    def __init__(self, input, n_in, n_out):

        # 设置权重W为theano共享变量并且初始化为0
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX),
            name='W',
            borrow=True)

        # 设置bias为0
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out, ),
                dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        # 求出每个样本的概率
        # P(y | x, theta) = softmax( x * W + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # 计算出样本的labels及属于哪类标签
        # labels = max(p_y_given_x)
        # p_y_given_x 对应为一个矩阵，每一行代表一个样本数
        # y_pred 为一个向量
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # 模型参数 theta
        self.params = [self.W, self.b]

        # 输入数据
        self.input = input


    # 定义 negative log likehood 函数用于计算模型
    # loss(theta) = - (1 / m) * sum(p(y|x, theta)) #m 为样本数
    def negative_log_likehood(self, y):
        # note : 在初始化函数中 p_y_given_x 计算出来的是一个矩阵(所有标签的概率),而在计算
        # loss的时候， p(y|x, theta)是一个向量，表示预测出该标签(y)的概率，而不是
        # 所有标签的概率
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # 定义误差函数用于validation 和 模型测试
    # 误差函数定义为: 错误分类的样本数的均值
    def errors(self, y):
        # 判断计算出来的y_pred 与 y 是否具有相同的维度
        if y.ndim != self.y_pred.ndim :
            raise (
                "y should have the same shape with y_pred" ,
                ('y', y.shape, 'y_pred', self.y_pred)
            )
        # 判断y是否是正确的类型
        elif  y.dtype.startswith('int') :
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

### 导入数据函数
### dataset 为数据集的路径
def load_data(dataset):

    # 判断数据集是否存在
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(data_file) :
        # 判断数据集是否在data目录下
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) and data_file == "mnist.pkl.gz" :
            dataset = new_path
    ## 如果数据集不再本地，则进行下载
    if (not os.path.isfile(dataset) and data_file == "mnist.pkl.gz") :
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print "Downloading data from %s" %origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    #将数据存入到共享变量中，提高性能
    def shared_dataset(data_xy, borrow = True) :
        data_x, data_y = data_xy
        shared_x = theano.shared(
            value=numpy.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow
        )
        shared_y = theano.shared(
            value=numpy.asarray(data_y, dtype=theano.config.floatX),
            borrow=borrow
        )

        # 由于mnist数据集是数字识别。。
        # 因此将labels y 保存为int型数据
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

## 对模型进行训练和建模。。求出参数
## 使用 minibatch stochastic gradient descent进行模型训练
## 使用early stopping 防止overfitting
def sgd_optimization_mnist(learning_rate = 0.13, n_epochs = 1000,
                           dataset='data/mnist.pkl.gz', batch_size=600) :

    ##################
    ## LOAD DATASET ##
    ##################

    ### 导入数据
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    ## 计算出对于每一个数据集 minibatch 的迭代次数
    ## 注意共享变量的取值
    n_train_batches = train_set_x.get_value().shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value().shape[0] / batch_size
    n_test_batches = test_set_x.get_value().shape[0] / batch_size

    ###########################
    ## BUILDING ACTUAL MODEL ##
    ###########################

    print "... Bulding Model"

    ## 定义模型tensortype 变量

    # 输入数据变量 x
    x = T.matrix('x')
    # 标签变量 y
    y = T.ivector('y')
    # minibatch index 变量
    index = T.iscalar('index')

    ### 创建一个Logistic Regression object
    ## MNIST 每个图片的大小为28 * 28
    ## labels 为 1 - 10
    classifier = LositicRegression(input=x, n_in=28 * 28, n_out=10)

    # 获取Logistic Regression Cost function
    # for training model
    cost = classifier.negative_log_likehood(y)

    ### 计算梯度 Gradients
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    ### 梯度计算
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    # 编译training model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # 编译validation Model
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size : (index + 1) * batch_size],
            y: valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # 编译test Model
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

    #使用early-stopping 防止overfitting
    print "... training model"
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequecy = min(patience / 2, n_train_batches)

    #初始化best_validation_loss
    best_validation_loss = numpy.inf
    test_score = 0

    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping) :
        epoch += 1
        ### minibatch stochastic gradient descent
        for minibatch_index in xrange(n_train_batches) :
            minibatch_avg_cost = train_model(minibatch_index)
            # 迭代次数
            iter = (epoch - 1) * n_train_batches + minibatch_index

            #判断是否需要验证
            if (iter + 1) % validation_frequecy == 0:
                # 计算出validation set 的0 - 1 loss
                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                #计算出此时模型的误差
                this_validation_losses = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i / %i, validation error %f %% ' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_losses
                    )
                )

                # 判断此时性能是否有改善
                if this_validation_losses < best_validation_loss:

                    # 如果性能有很大的提升，则update patience
                    if this_validation_losses < best_validation_loss * improvement_threshold :
                        patience = max( patience, iter * patience_increase)

                    best_validation_loss = this_validation_losses

                    ## 在测试集上测试此模型
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print (
                        '      epoch %i, minibatch %i / %i, test error of '
                        'best model %f %%' %
                        (
                            epoch, minibatch_index, n_train_batches, test_score * 100
                        )
                    )

                    # 保存此时模型的参数
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

                if patience <= iter :
                    done_looping = True
                    break

    end_time = timeit.default_timer()
    print (
        "Optimization complete with best validation score of %f %%,"
        "with test performance %f %%."
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epoches / sec ' %(
        epoch, 1. * epoch / (end_time - start_time)
    )

    print >> sys.stderr, (' The code for file ' +
                          os.path.split(__file__)[1] +
                          'ran for %.1fs' % ((end_time - start_time)))

## 如果模型建好了，可以使用该模型进行预测
def predict() :

    ## 导入保存好的模型参数
    classifier = cPickle.load(open('best_model.pkl', 'rb'))

    ## 编译预测模型
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=[classifier.y_pred]
    )

    ## 可以使用测试数据集对该模型进行预测
    datasets = load_data('data/mnist.pkl.gz')
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()


    predict_values = predict_model(test_set_x[30:50, :])
    print "predict values: ", predict_values
    #print "actual  values: ", test_set_y[30:50]

if __name__ == "__main__":
    #sgd_optimization_mnist()
    predict()

