import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy

import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared(
    value=numpy.asarray(
        rng.uniform(
            low=-1.0/w_bound,
            high=1.0/w_bound,
            size=w_shp),
        dtype=input.dtype),
    name='w'
)

b_shp=(2,)
b = theano.shared(
    value=numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shp),
                        dtype=input.dtype
    ),
    name='b'
)

# build symbolic expression that commputes teh convolution of input with filters in w
conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))


# create theano function to compute filtered images
f = theano.function([input], output)

# open random image of dimensnions 639 x 516
img = Image.open(open('data/3wolfmoon.jpg'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape(1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_image = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();

# recall that the convOp output (filtered image) is actually a 'minibatch',
# of size 1 here, so we take index 0 in the first dimension
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_image[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_image[0, 1, :, :])
pylab.show()


input = T.dtensor4('input')
maxpool_shape = (2,2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
fd = theano.function([input], pool_out)

invals = numpy.random.RandomState(1).rand(3,2,5,5)
print 'With ignore_border set to True: '
print 'invals[0,0,:,:] = \n', invals[0,0,:,:]
print 'output[0,0,:,:] =\n', fd(invals)[0,0,:,:]

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
ff = theano.function([input], pool_out)
print 'with ignore_border set to False: '
print 'invals[1,0,:,:]=\n', invals[1,0,:,:]
print 'output[1,0,:,:]=\n', ff(invals)[1,0,:,:]

