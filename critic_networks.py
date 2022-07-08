#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:12:09 2022

@author: c
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import logging
logging.disable(logging.WARNING)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAvgPool2D, Conv2D, LeakyReLU
from tensorflow.keras.layers import LayerNormalization, Activation, Layer
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
#%%

"""
In addition to the commonly used LayerNormalization and InstanceNormalization
layers, we implement the PixelNormalization layer and the Identity layer, where
the latter one is a helper layer if NO normaliztion is considered. """
	
# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
	# initialize the layer
	def __init__(self,scale=False, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate square pixel values
		values = inputs**2.0
		# calculate the mean pixel values
		mean_values = K.mean(values, axis=-1, keepdims=True)
		# ensure the mean is not zero
		mean_values += 1e-8
		# calculate the sqrt of the mean squared value (L2 norm)
		l2 = K.sqrt(mean_values)
		# normalize values by the l2 norm
		normalized = inputs / l2
		return normalized
 
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		return input_shape
    
    
class Identity(Layer):
	# initialize the layer
	def __init__(self,scale=False, **kwargs):
		super(Identity, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):		
		return inputs
    
#%%
    

class Conv2DHE(Conv2D):
    """
    Standard Conv2D layer but includes He normal initialization of the
    convolution kernels.
    """
    def __init__(self,dropout=False, **kwargs):
        self.dropout = dropout
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(
            kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)

    def call(self, inputs):
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,#*self.c, # scale kernel
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.dropout:
            outputs = K.dropout(outputs,.2)
            
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
    
class Conv2DEQ(Conv2D):
    """
    Standard Conv2D layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Conv2D layer and overrides the call method, following
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    """
    def __init__(self,dropout=False, **kwargs):
        self.dropout = dropout
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=1), **kwargs)
        # super().__init__(
        #     kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        #n = np.product([int(val) for val in input_shape[1:]])
        shape = tf.shape(self.kernel)
        n = np.product([int(val) for val in shape[:3]])
        # He initialisation constant
        self.c = np.sqrt(2/n)

    def call(self, inputs):
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,#*self.c, # scale kernel
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.dropout:
            outputs = K.dropout(outputs,.2)
            
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
    
class DenseHE(Dense):
    """
    Standard dense layer but includes He normal initialization of the weights.
    """
    def __init__(self, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(
            kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    
class DenseEQ(Dense):
    """
    Standard dense layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Dense layer and overides the call method.
    """
    def __init__(self, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=1), **kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel*self.c) # scale kernel
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    

#%%

# define the PatchGAN critic

def define_patchgan_critic(in_image,f,norm,k = 4,alpha=0.2,out_act='linear'):
    
    f=int(f); 
    
    if norm=='none':
        cNormalization = Identity
    elif norm=='layer':
        cNormalization = LayerNormalization
    elif norm=='instance':
        cNormalization = InstanceNormalization
    else:
        print('No valid norm defined'); pass;
    
    d = Conv2DHE(filters=f,kernel_size=(k,k),padding='same',strides=2)(in_image)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DHE(filters=2*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DHE(filters=4*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=8*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=8*f,kernel_size=(k,k),padding='same',strides=1)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=1, kernel_size=(k,k), padding='valid',strides=1)(d)
    patch_out = Activation(out_act)(d)
    model = Model(in_image, patch_out)
   
    return model

###############################################################################

# this helper function can be used to check the receptive field of the critic
def calculate_respective_field(S,F):
    r=1
    for l in range(1,len(S)+1,1):
        s=1
        i=1
        while i<l:
            s*=S[i-1]
            i+=1
        r+=(F[l-1]-1)*s
    return r

calculate_respective_field([2,2,2,2,1,1],[4,4,4,4,4,4])

###############################################################################

# define a DCGAN critic

def define_dcgan_critic(in_image,f,norm,k = 4,alpha=0.2,out_act='linear'):
    
    f=int(f);
    
    if norm=='none':
        cNormalization = Identity
    elif norm=='layer':
        cNormalization = LayerNormalization
    elif norm=='instance':
        cNormalization = InstanceNormalization
    else:
        print('No valid norm defined'); pass;
    
    d = Conv2DHE(filters=f,kernel_size=(k,k),padding='same',strides=2)(in_image)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DHE(filters=2*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DHE(filters=4*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=8*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=16*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DHE(filters=1, kernel_size=(k,k), padding='valid',strides=1)(d)
    dcgan_out = Activation('linear')(d)
    
    model = Model(in_image, dcgan_out)
   
    return model





#%%

