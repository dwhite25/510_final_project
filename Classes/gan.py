'''
houses main classes and functions for MSG-GAN architecture for time series.
'''

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import math
import csv
import sys
from pathlib import Path
from functools import partial
from tensorflow import keras as K
from tensorflow.keras import Model, backend
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv1D, Conv2DTranspose, Conv1DTranspose
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, UpSampling2D, UpSampling1D
from tensorflow.keras.layers import Concatenate, LeakyReLU, Reshape, Input, Dot, Multiply
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam, RMSprop, schedules
from tensorflow.keras.initializers import RandomNormal


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class DenseEQ(Dense):
    def __init__(self, channels, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=1), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        output = backend.dot(inputs, self.kernel*self.c) # scale kernel
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias*self.c)
        if self.activation is not None:
            output = self.activation(output)
        return output


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# custom version of Conv1D that uses equalized learning rate
class Conv1DEQ(Conv1D):
    def __init__(self, channels, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=1), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        outputs = tf.nn.conv1d(inputs, self.kernel*self.c, stride=self.strides,
                 padding='SAME', data_format='NWC')

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias*self.c, data_format='NWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Conv1DTransposeEQ(Conv1DTranspose):
    def __init__(self, channels, out_shape, **kwargs):
        self.channels  = channels
        self.out_shape = out_shape
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=1), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        outputs = tf.nn.conv1d_transpose(inputs, self.kernel*self.c, self.out_shape, 
                strides=self.strides, padding='SAME', data_format='NWC')

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias*self.c, data_format='NWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class MiniBatchStdDevTs(Layer):
    def __init__(self, **kwargs):
        super(MiniBatchStdDevTs, self).__init__(**kwargs)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        mean = backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = backend.square(inputs - mean)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)
        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], 1)) 
        combined = backend.concatenate([inputs, output], axis=-1)
        
        return combined 


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# MSG_GAN for time series. 
class MSG_GAN_ts:
# ---------------------------------------------------------------------------------------------------
    def __init__(self, neurs=512, epochs=50000, batch_size=32, g_lr=1e-4, d_lr=1e-4, r1_gamma=10.0, 
                 epsilon=1e-3, eps2=1e-7, beta1=0.5, beta2=0.999, rho=0.9, momentum=0.0, endres=128,
                 outpath='', nchannels=1, ksize=5, usebias=False, startres=4, nlabels=4, alt=0, 
                 ellog=True, amsgrad=False, **kwargs):
        self.neurs      = neurs
        self.nlabels    = nlabels
        self.epochs     = epochs
        self.batch_size = batch_size
        self.outpath    = outpath
        self.g_lr       = g_lr
        self.d_lr       = d_lr
        self.decay      = 0.0
        self.r1_gamma   = r1_gamma
        self.epsilon    = epsilon
        self.eps2       = eps2
        self.beta1      = beta1
        self.beta2      = beta2
        self.rho        = rho
        self.momentum   = momentum
        self.startres   = startres
        self.endres     = endres
        self.nchannels  = nchannels
        self.kernels    = 0
        self.ksize      = ksize
        self.to_ts      = []     
        self.reals      = []
        self.usebias    = usebias
        self.alt        = alt
        self.ellog      = ellog
        self.amsgrad    = amsgrad
        self.g_opt      = self.declare_g_optimizer(opt='adam')   # (no argument) defaults to RMSProp. 
        self.d_opt      = self.declare_d_optimizer(opt='adam')   # we could add other optimizers later
        self.G          = self.generator()
        self.D          = self.discriminator()

        # self.G.summary()
        # self.D.summary()

 # -----------------------------------------------------------------------------------------------
    # the generator model
    def generator(self):
        self.kernels    = 0
        iters           = int(math.log2(self.endres)) - int(math.log2(self.startres))
        channels        = self.neurs
        self.to_ts      = []
        inputs          = []

        inputs.append(Input(shape=(self.nlabels,), name='labels')) 

        z = inputs[0]
        z = Reshape((1, self.nlabels))(z)

        x = DenseEQ(channels, activation=LeakyReLU(0.2), use_bias=self.usebias)(z)

        x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, self.startres, channels], 
                kernel_size=self.startres, strides=self.startres, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                kernel_size=min(self.ksize, 5), strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = self.pixel_norm(x)

        for block in range(iters):
            z = Conv1DTransposeEQ(1, out_shape=[self.batch_size, x.shape[1], 1], 
                    kernel_size=1, strides=1, use_bias=self.usebias, activation=LeakyReLU(0.2))(x)
            self.to_ts.append(z)

            x = UpSampling1D()(x)

            for i in range(2):
                x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                        kernel_size=self.ksize, strides=1, activation=LeakyReLU(0.2),
                        use_bias=self.usebias)(x)
                x = self.pixel_norm(x)

        x = Conv1DTransposeEQ(1, 
                out_shape=[self.batch_size, x.shape[1], 1], 
                kernel_size=1, strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)

        # put to_ts in backwards order for the discriminator
        ts2 = []
        for i in range(len(self.to_ts)):
            ts2.append(self.to_ts[-(i+1)])
        self.to_ts = ts2
        
        outs = [x]
        for i in self.to_ts:
            outs.append(i)
        self.kernels = channels

        return Model(inputs, outs, name='Generator')

    # -----------------------------------------------------------------------------------------------
    # the discrimnator model
    def discriminator(self):
        iteration   = 1                     # tracks which batch of minis we should be feeding in
        inputs      = []
        channels    = self.kernels
        iters       = int(math.log2(self.endres)) - int(math.log2(self.startres))

        inputs.append(Input(shape=(self.endres, self.nchannels), name='max_res'))
        x = inputs[0]
        
        cntr = 0
        for i in self.to_ts:
            cntr += 1
            inputs.append(Input(shape=i.shape[1:], ragged=True, name=('downscale_%d' %(cntr))))
        inputs.append(Input(shape=(self.nlabels,), ragged=True, name='labels'))
        x = Conv1DEQ(channels, kernel_size=1, strides=1, activation=None)(x)
        x = MiniBatchStdDevTs()(x)


        for block in range(iters):
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            x = AveragePooling1D(strides=2)(x)
            x = Concatenate(axis=-1)([x, inputs[iteration][:,:,:1]])
            x = MiniBatchStdDevTs()(x)
            
            iteration += 1

        assert channels == self.neurs, 'channels do not == self.neurs at end of discriminator'
        x = Conv1DEQ(channels, kernel_size=min(self.ksize, 5), strides=1, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)
        x = Conv1DEQ(channels, kernel_size=self.startres, strides=self.startres, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)

        z = inputs[-1]
        z = Reshape((1, self.nlabels))(z)

        x = Concatenate(axis=-1)([x, z])
        x = DenseEQ(1, activation='linear', use_bias=self.usebias)(x)

        return Model(inputs, x, name='Discriminator')


    # -----------------------------------------------------------------------------------------------
    # main training loop
    def train(self, reals):
        print('starting training loop')
        print('')
        self.batch_size = reals.batch_size
        self.endres     = reals.endres
        self.nchannels  = reals.nchannels

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   

        g_train_loss    = Mean()
        d_train_loss    = Mean()

        assert self.endres != 0, 'the size of endres was not correctly stored'
        assert self.nchannels != 0, 'the number of channels was not correctly stored'

        for epoch in range(self.epochs):
            reals.create_batch()
            d_loss, d_grad  = self.train_d(reals)
            g_loss, g_grad  = self.train_g(reals)
            if epoch % 10 == 0:
                self.print_losses(epoch, g_loss, g_train_loss, g_grad, d_loss, d_train_loss, d_grad)

            if epoch % 100 == 0:
                outputs = self.generate_samples(reals)
                reals.report_objects(outputs[0], epoch)
                if ((epoch % 1000 == 0) and (epoch > 0)):
                    self.G.save_weights(self.outpath + '/weights/generator%05d.h5' %(epoch), True)
                    self.D.save_weights(self.outpath + '/weights/discriminator%05d.h5' %(epoch), True)

       # -----------------------------------------------------------------------------------------------
    @tf.function
    def train_g(self, reals):
        # noise = np.random.normal(size=[self.batch_size, self.neurs-self.nlabels], loc=0.5, scale=0.15)
        labels = []
        if self.nlabels != 0:
            labels_g = np.array([self.prepare_labels(i) for i in reals.labels_batch], dtype=np.float32)
        with tf.GradientTape() as t:
            fakes = self.G(labels_g, training=True)
            yfake = self.D([fakes, labels_g], training=True)
            loss  = self.get_g_loss(yfake)
        grads = t.gradient(loss, self.G.trainable_variables)
        goods = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in grads]))
        tf.cond(goods, lambda: self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables)), 
                                                                                false_fn=tf.no_op)

        return loss, grads

    # -----------------------------------------------------------------------------------------------
    @tf.function
    def train_d(self, reals):
        # noise = np.random.normal(size=[self.batch_size, self.neurs-self.nlabels], loc=0.5, scale=0.15)
        labels = []
        if self.nlabels != 0:
            labels_g = np.array([self.prepare_labels(i) for i in reals.labels_batch], dtype=np.float32)
            labels_d = np.array([self.prepare_labels(i) for i in reals.labels_batch], dtype=np.float32)
        self.reals = self.make_tensors(reals.objects)
        with tf.GradientTape() as t:
            fakes = self.G(labels_g, training=True)
            yfake = self.D([fakes, labels_g], training=True)
            yreal = self.D([self.reals, labels_d], training=True)
            loss  = self.get_d_loss(yreal, yfake, self.reals, fakes, labels_g, labels_d)
        grads = t.gradient(loss, self.D.trainable_variables)
        goods = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in grads]))
        tf.cond(goods, lambda: self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables)), 
                                                                                false_fn=tf.no_op)

        return loss, grads


    # -----------------------------------------------------------------------------------------------
    def get_g_loss(self, yfake):
        assert yfake is not None

        return -yfake

    # -----------------------------------------------------------------------------------------------
    # applies a unique gp per-resolution rather than a single gp applied to all gradients
    def get_d_loss(self, yreal, yfake, reals, fakes, labels_g, labels_d):
        assert yfake is not None
        assert yreal is not None
        # reals   = [tf.cast(reals[i], tf.float32) for i in range(np.shape(reals)[0])]
        alpha1  = tf.random.uniform([int(self.batch_size), 1, 1], 0., 1.)
        alpha2  = tf.random.uniform([int(self.batch_size), 1], 0., 1.)
        loss    = yfake - yreal
        inter1  = [reals[i] + (alpha1 * (fakes[i] - reals[i])) for i in range(len(reals))]
        inter2  = labels_d + (alpha2 * (labels_g - labels_d))
        discr   = partial(self.D, training=True)
        with tf.GradientTape() as t:
            t.watch(inter1)
            t.watch(inter2)
            pred = discr([inter1, inter2])
        grads   = t.gradient(pred, [inter1, inter2])[0]
        slopes  = [tf.sqrt(tf.reduce_sum(tf.square(grads[i]), axis=[1, 2])) 
                                            for i in range(len(reals))]
        gp      = [tf.reduce_mean((slopes[i] - 1.)**2) for i in range(len(reals))]
        ep      = [tf.square(yreal[i]) for i in range(len(reals))]
        for i in range(len(reals)):
            loss   += gp[i]*self.r1_gamma + ep[i]*self.epsilon

        return loss

    # -----------------------------------------------------------------------------------------------
    # modifies labels with gaussian offsets and saves objects
    def make_tensors(self, reals):
        tensrs = [[] for i in range(np.shape(reals)[1])]
        for sim in reals:
            for lvl in range(np.shape(sim)[0]):
                tensrs[lvl].append(np.array(sim[lvl], dtype=np.float32)) 
        tensrs = [tf.constant(np.array(i)) for i in tensrs]
        
        return tensrs

    # -----------------------------------------------------------------------------------------------
    def print_losses(self, epoch, g_loss, g_train_loss, g_grad, d_loss, d_train_loss, d_grad):
        g_zeroes = np.mean([tf.math.zero_fraction(g_grad[i]) for i in range(np.shape(g_grad)[0])])
        d_zeroes = np.mean([tf.math.zero_fraction(d_grad[i]) for i in range(np.shape(d_grad)[0])])
        g_train_loss(tf.math.abs(g_loss))
        d_train_loss(tf.math.abs(d_loss))
        if epoch == 0:
            print('LOSSES:')
            print('# ---------------------------------------------------------')
            Path(self.outpath).mkdir(exist_ok=True)
            Path(self.outpath + '/weights').mkdir(exist_ok=True)
        print('epoch %d:   Generator: %f    Discriminator: %f' 
                %(epoch, g_train_loss.result(), d_train_loss.result()))
        with open(self.outpath + '/weights/losses.csv', 'a', newline='') as f:
	        writer = csv.writer(f)
	        writer.writerow([int(epoch), float(g_train_loss.result()), float(d_train_loss.result())])

    # -----------------------------------------------------------------------------------------------
    # this can be overridden in subclasses if we find different architectures require different opts
    def declare_g_optimizer(self, opt=None):
        if opt == 'adam':
            return Adam(self.g_lr, self.beta1, self.beta2, self.eps2, amsgrad=self.amsgrad)
        elif opt == 'adam-wd': 
            lr = self.d_lr
            wd = lambda: self.decay
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, beta_1=self.beta1,
                                        beta_2=self.beta2, epsilon=self.eps2)
        else:
            return RMSProp(self.g_lr, self.rho, self.momentum, self.eps2)

    # -----------------------------------------------------------------------------------------------
    # this can be overridden in subclasses if we find different architectures require different opts
    def declare_d_optimizer(self, opt=None):
        if opt == 'adam':
            return Adam(self.d_lr, self.beta1, self.beta2, self.eps2, amsgrad=self.amsgrad)
        elif opt == 'adam-wd': 
            lr = self.d_lr
            wd = lambda: self.decay
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, beta_1=self.beta1,
                                        beta_2=self.beta2, epsilon=self.eps2)
        else:
            return RMSProp(self.d_lr, self.rho, self.momentum, self.eps2)

    # -----------------------------------------------------------------------------------------------
    def pixel_norm(self, x, epsilon=1e-8):
        epsilon = tf.constant(epsilon, dtype=x.dtype)

        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))
   
    # -----------------------------------------------------------------------------------------------
    # 'lbls' should be an array with [freq, amp] in that order
    def makewaves(self, reals, epoch, lbls):
        outputs = self.generate_waves(lbls)
        reals.save_objects(outputs[0], lbls, epoch, self.outpath)

    # -----------------------------------------------------------------------------------------------
    # not ideal for creating single waveforms. might have to think through that part later
    @tf.function
    def generate_samples(self, reals):
        # noise  = np.random.normal(size=[self.batch_size, self.neurs-self.nlabels], loc=0.5, scale=0.15)
        labels = np.array([self.prepare_labels(i) for i in reals.labels_batch], dtype=np.float32)

        return self.G(labels, training=False)

    # -----------------------------------------------------------------------------------------------
    # not ideal for creating single waveforms. might have to think through that part later
    def generate_waves(self, lbls):
        # noise  = np.random.normal(size=[self.batch_size, self.neurs-self.nlabels], loc=0.5, scale=0.15)
        labels = np.array([self.prepare_labels(lbls, sd=False) for i in range(32)], dtype=np.float32)

        return self.G(labels, training=False)

    # -----------------------------------------------------------------------------------------------
    def prepare_labels(self, labels, sd=True):
        # labels are in following order: [freq, amp]
        freq  = labels[0]
        amp   = labels[1]
        stddev = 0.00

        if sd == True:
            # add in noise to labels to simulate continuous parameter space
            freq  += np.random.normal(0.0, stddev)
            amp   += np.random.normal(0.0, stddev)

        # scale values to [0, 1], without being too close to either 0 or 1
        # amplitude was already scaled at time of file creation
        freq /= 125 + 0.1         # freq is between 1 and 100

        # return new array of labels
        return [freq, amp]