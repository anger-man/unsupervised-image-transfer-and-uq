#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:02:52 2022

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
from random import random
from numpy import zeros
from numpy import ones
from numpy import asarray
import math
from numpy.random import randint
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAvgPool2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2D, SeparableConv2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Lambda, ReLU
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, AveragePooling2D, LayerNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from matplotlib.image import imread
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import optparse
from scipy.ndimage import zoom
import skimage.measure
import scipy
from skimage.metrics import structural_similarity as ssim
#%%

def psnr(y_true,y_pred):
    res = 20*np.log10(np.max(y_true))-10*np.log10(np.mean(np.square(y_true-y_pred)))
    return res

def load_data(task,number_of_faces, index, inp, tar,domain,grayscale=False):
    X = []
    for k in range(number_of_faces):
        i1 = (index+k)%len(inp)
        i2 = (index+k)%len(tar)
        if domain == 'input':
            x = imread(inp[i1]); 
            x = (x-127.5)/127.5
        else:
            x = imread(tar[i2])
        X.append(x.astype(np.float32))
    X=np.array(X)
    # if grayscale images
    if len(X.shape)!=4:
        X = np.expand_dims(X,-1)
    if domain=='target':
        X = (X-127.5) / 127.5
    if grayscale:
        X = .299*X[...,0:1] + .587*X[...,1:2] + .114*X[...,2:3]
    
    if task in ['tex/','dog/','dt4/']:
        X = X[:,::2,::2]
    return (X)

def validate(task,dD1,dD2,gCtoD,evaluation,dim,MyFilter,coord):
    X1 = [];X2=[]
    try:
        samples = np.random.choice(range(len(evaluation)),64,replace=False)
    except:
        samples = range(len(evaluation))
        
    evinp = np.array(evaluation['input'])[samples]
    evtar = np.array(evaluation['target'])[samples]
    
    X1 = load_data(task,len(samples),0,evinp,evtar,domain='input')
    X1 = scale_data(X1,dim)
    X2 = load_data(task,len(samples),0,evinp,evtar,domain='target')
    X2 = scale_data(X2,dim)
    # if grayscale images
    coord_real=[]
    for dummy in range(X1.shape[0]):
        pwidth=(np.random.uniform(.7,1.)*dim[0]).astype(np.int32)
        coord_real.append(pwidth)
        coord_real.append(np.random.choice(range(dim[0]-pwidth)))
        coord_real.append(np.random.choice(range(dim[0]-pwidth)))
   
    
    x_realDori = scale_data(X2,dim)
    
    x_realClpf = MyFilter(coord_real)(X1)
    x_realDlpf = MyFilter(coord_real)(x_realDori)
    
    #do not increase predict batch_size -> resource exhausted error for dim>128
    x_fakeDori = gCtoD.predict(X1,    batch_size=2)[...,:X2.shape[-1]]
    x_fakeDlpf = gCtoD.predict(x_realClpf, batch_size=2)[...,:X2.shape[-1]]
    loss1 = np.mean(dD1.predict(x_realDori,batch_size=2)-dD1.predict(x_fakeDori,batch_size=2))
    loss2 = np.mean(dD2.predict(x_realDlpf,batch_size=2)-dD2.predict(x_fakeDlpf,batch_size=2))
    return(loss1,loss2)

avg_pool = AveragePooling2D(pool_size=(2,2),padding='valid',strides=(1,1))

def evall_2(step, gCtoD,evaluation,name,task,dim,paired=False):
    X1 = [];X2=[]; dif1=0
    e_inp = evaluation['input']; e_tar = evaluation['target']
    
    n_samples = 512
    try:
        samples = np.random.choice(range(np.min([len(e_inp),len(e_tar)])),n_samples,replace=False)
    except:
        samples = np.random.permutation(np.min([len(e_inp),len(e_tar)]))
    
    
    evinp = np.array(evaluation['input'])[samples]
    evtar = np.array(evaluation['target'])[samples]
    
    X1 = load_data(task,len(samples),0,evinp,evtar,domain='input')
    X1 = scale_data(X1,dim)
    X2 = load_data(task,len(samples),0,evinp,evtar,domain='target')
    X2 = scale_data(X2,dim)
   
    preds_B = gCtoD.predict(X1, batch_size=2)[...,:X2.shape[-1]]
    X1 = (X1+1) * 127.5    
    if task in ['tex/']:
        preds_B = (preds_B+1)/2
        X2 = (X2+1)/2
    if task in ['ixi/','rir/','kne/','hea/']:
        preds_B = (preds_B+1)*127.5
        X2 = (X2+1)*127.5
    if task in ['dt4/']:
        preds_B = preds_B*5
        X2 = X2*5
        preds_B = avg_pool(preds_B)
        X2 = avg_pool(X2)
    if task in ['sur/']:
        preds_B = preds_B*0.4725
        X2 = X2*0.4725
    if task in ['dog/']:
        preds_B = (preds_B+1) * 127.5  
        preds_B = preds_B.astype(np.uint8)
        X2 = (X2+1) * 127.5  
        X2 = X2.astype(np.uint8)
        
    fig = plt.figure(figsize=(5,6.5))
    for i in range(5):
        ax=fig.add_subplot(6, 5, 1 + i)
        plt.axis('off')
        ax.imshow(X1[i].astype(np.uint8))
    for i in range(5):
        ax=fig.add_subplot(6, 5, 5+1 + i)
        plt.axis('off')
        ax.imshow(preds_B[i])
    for i in range(5):
        ax=fig.add_subplot(6, 5, 2*5+1 + i)
        plt.axis('off')
        ax.imshow(X2[i])
    for i in range(5):
        ax=fig.add_subplot(6, 5, 3*5+1 + i)
        plt.axis('off')
        ax.imshow(X1[i+5].astype(np.uint8))
    for i in range(5):
        ax=fig.add_subplot(6, 5, 4*5+1 + i)
        plt.axis('off')
        ax.imshow(preds_B[i+5]);
    for i in range(5):
        ax=fig.add_subplot(6, 5, 5*5+1 + i)
        plt.axis('off')
        ax.imshow(X2[i+5]);
    
    if paired:
        if task in ['tex/','dt4/','sur/']:
            dif1 = -np.mean(np.square(X2-preds_B))**.5
            dif2 = -np.mean(np.abs(X2-preds_B))
            fig.suptitle('RMSE: %.3f\n' %(-dif1))
        else:
            dif1 = ssim(X2,preds_B,multichannel=True,data_range=X2.max()-X2.min())*100
            dif2 = psnr(X2,preds_B)
            fig.suptitle('SSIM: %.3f\n' %dif1)
            
    else:
        dif1=0; dif2=0
        
    filename ='plots/%s_%04d.jpg' % (name,(step+1))
    fig.tight_layout(pad=.1)
    plt.savefig( filename,dpi=200)

    return(dif1,dif2)


#%%

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif np.random.uniform() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = np.random.randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def make_fid(task,evaluation,stage_dims,STAGE,count,patches,out_dim,gCtoD,inc_model):
    sd=stage_dims[STAGE][0]
    images1 = generate_real_samples(load_data(task,128,count,evaluation['input'],evaluation['target'],
                                            domain='input'), patches)[0]
    images1 = scale_data(images1, stage_dims[STAGE])
    images1 = generate_fake_samples(gCtoD, images1, patches,out_dim[-1])[0]
    images1 = images1[:,int(.2*sd):int(.8*sd),int(.2*sd):int(.8*sd),:]
    images1 = scipy.ndimage.zoom(images1, (1,299/images1.shape[1],299/images1.shape[2],1), order=1)
    
    images2 = generate_real_samples(load_data(task,128,count,evaluation['input'],evaluation['target'],
                                            domain='target'), patches)[0]
    images2 = scale_data(images2, stage_dims[STAGE])
    images2 = images2[:,int(.2*sd):int(.8*sd),int(.2*sd):int(.8*sd),:]
    images2 = scipy.ndimage.zoom(images2, (1,299/images2.shape[1],299/images2.shape[2],1), order=1)
    # images2 /= np.max([1,np.max(np.abs(images2),axis=(1,2,3))])
    
    images1 = np.concatenate([images1,images1,images1], axis=-1)
    images2 = np.concatenate([images2,images2,images2], axis=-1)
    dif1 = calculate_fid(inc_model, images2, images2)
    dif2 = calculate_fid(inc_model, images1, images2)
    
    return dif1, dif2

def scale_data(x,dim):
    shape = x.shape
    res = np.zeros((shape[0],dim[0],dim[1],shape[-1]))
    if dim[0]<shape[1]:
        fac = int(shape[1]/dim[0]);
        for k in range(shape[0]):
            res[k,...] =  skimage.measure.block_reduce(x[k], (fac,fac,1), np.mean)
    else:
        fac = int(dim[0]/shape[1])
        res = x.repeat(fac, axis=1).repeat(fac, axis=2)
    return(res.astype(np.float32))

def generate_fake_samples(generator, dataset, patch_shape, out_channel):
    # generate fake instance
    X = generator.predict(dataset,batch_size=2)[...,:out_channel]
    # X = np.concatenate((X[0],X[1],X[2]),axis=0)
    # create 'fake' class labels (0)
    try:
        y = ones((len(X), patch_shape[0], patch_shape[1], 2))
    except:
        y = ones((len(X), 1))
    return X, y

def generate_real_samples(dataset, patch_shape):
    X = dataset
    # generate 'real' class labels (1)
    try:
        y = -ones((len(X), patch_shape[0], patch_shape[1], 2))
    except:
        y = -ones((len(X), 1))
    return X, y

def plot_curves_gp(dB,gB,alpha,dDval,criticW,name):
    dB=np.array(dB); gB = np.array(gB)
    dDval = np.array(dDval)
    
    plt.figure()
    l2 = np.convolve(dB[80:],np.ones(20)/20,mode='valid')
    b,=plt.plot(l2); 
    l3 = np.array(dDval[:,0])
    l4 = np.array(dDval[:,1])
    c,=plt.plot(np.convolve(l3[80:],np.ones(20)/20,mode='valid'),linewidth=.3);
    d,=plt.plot(np.convolve(l4[80:],np.ones(20)/20,mode='valid'),linewidth=.3); 
    plt.legend((b,c,d),('W1_D','W1_Dval','W1_DvalROT'))
    plt.title(str('W1_D: %.2f' %(np.mean(l2[max(len(l2)-100,0):]))))
    plt.savefig('losses/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4)) 
    l2 = np.mean(gB[80:,1:3],axis=-1)
    b,=plt.plot(np.convolve(l2,np.ones(20)/20,mode='valid')); 
    l3 = alpha*np.sum(gB[80:,3:], axis=-1)
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    plt.legend((b,c),('gen','rec'))
    plt.savefig('CtoBlosses/%s.pdf' %name,dpi=200)
    



class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=0.005):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
    

    

###############################################################################
      
def resnet_block(f,input_layer,red = False, norm= True):
    shortcut = input_layer
    if red:
        g = Conv2D(f, (3,3), padding='same', strides =(2,2))(input_layer)
        shortcut = Conv2D(f, (1,1), padding='same', strides=(2,2))(shortcut)
        if norm:
            shortcut = GroupNormalization(groups=min(f,norm))(shortcut)
    else:
        g = Conv2D(f, (3,3), padding='same')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Add()([g, shortcut])
    g = Activation('relu')(g)
    return g


def up_block(f, input_layer, skip, norm = True):
    g = Conv2D(f, (4,4), strides=(1,1), padding='same')(input_layer)
    if norm:
        g = GroupNormalization(groups=f)(g)
    g = Activation('elu')(g)
    g = UpSampling2D(size=(2,2),interpolation='nearest')(g)
    g = Concatenate()([g,skip])
    g = Conv2D(f, (3,3), padding='same',strides=(1,1))(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('elu')(g)   
    return g


    
def residual_block(f, input_layer,norm=True):
    g = Conv2D(f, (3,3), padding='same')(input_layer)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)
    g = Conv2D(f, (3,3), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Concatenate()([g, input_layer])
    return g



###############################################################################


def define_style_transfer(inp_dim,out_dim,f,norm, out_act = 'tanh'):
    f = int(f); norm = int(norm)

    in_image = Input(shape=inp_dim)
    g = Conv2D(f, (7,7), padding='same')(in_image)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(2*f, (3,3), strides=(2,2), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(4*f, (3,3), strides=(2,2), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    for kkk in range(9):
        g = residual_block(4*f, g,norm=norm)

    g = Conv2DTranspose(2*f, (3,3), strides=(2,2), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2DTranspose(f, (3,3), strides=(2,2), padding='same')(g)
    if norm:
        g = GroupNormalization(groups=min(f,norm))(g)
    g = Activation('relu')(g)

    g = Conv2D(out_dim[-1], (7,7), padding='same')(g)
    out_image = Activation(out_act)(g)

    model = Model(in_image, out_image)
    return model
    

	
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
        # super().__init__(
        #     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=1), **kwargs)
        super().__init__(
            kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     # The number of inputs
    #     #n = np.product([int(val) for val in input_shape[1:]])
    #     shape = tf.shape(self.kernel)
    #     n = np.product([int(val) for val in shape[:3]])
    #     # He initialisation constant
    #     self.c = np.sqrt(2/n)

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
    
class DenseEQ(Dense):
    """
    Standard dense layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Dense layer and overides the call method.
    """
    def __init__(self, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        #super().__init__(
            #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=1), **kwargs)
        super().__init__(
            kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     # The number of inputs
    #     n = np.product([int(val) for val in input_shape[1:]])
    #     # He initialisation constant
    #     self.c = np.sqrt(2/n)

    def call(self, inputs):
        #output = K.dot(inputs, self.kernel*self.c) # scale kernel
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    
gNormalization = InstanceNormalization
def unet_enc(f, input_layer, red=True, norm=True, dr=True):
    f = np.min([f,512])
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(input_layer)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    shortcut = g
    g = MaxPooling2D((2,2))(g)
    return(g,shortcut)

def unet_dec(f, input_layer, skip, short=True, norm = True, dr=True):
    g = UpSampling2D((2,2),interpolation='nearest')(input_layer)
    g = Conv2DEQ(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,skip])
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    return(g)




def define_unet(inp_dim,out_dim,f, norm = 0, out_act = 'tanh',dr=True):
    f=int(f); norm=int(norm)
    
    in_image = Input(shape=inp_dim)
    g = in_image
    g = Conv2DEQ(filters=f,kernel_size=(3,3), padding = 'same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g, g_0 = unet_enc(f,    g, red= True, norm=norm,dr=dr)
    g, g_1 = unet_enc(2*f,  g, red= True, norm=norm,dr=dr)
    g, g_2 = unet_enc(4*f,  g, red= True, norm=norm,dr=dr)
    g, g_3 = unet_enc(8*f,  g, red= True, norm=norm,dr=dr)
    g, g_4 = unet_enc(16*f, g, red= True, norm=norm,dr=dr)
    foo, g = unet_enc(32*f, g, red= True, norm=norm,dr=dr)
    # foo, g = unet_enc(32*f, g, red= True, norm=norm)
    # g = unet_dec(32*f,g, g_5, norm=norm)
    g = unet_dec(16*f,g, g_4, norm=norm,dr=dr)
    g = unet_dec(8*f, g, g_3, norm=norm,dr=dr)
    g = unet_dec(4*f, g, g_2, norm=norm,dr=dr)
    g = unet_dec(2*f, g, g_1, norm=norm,dr=dr)
    foo = UpSampling2D((2,2),interpolation='nearest')(g)
    
    g = Conv2DEQ(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,g_0])
    # g = Activation('linear')(g)
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_image = Conv2DEQ(filters=out_dim[-1], kernel_size=(3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    
    g = Conv2DEQ(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,g_0])
    # g = Activation('linear')(g)
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_uncer = Conv2DEQ(filters=1, kernel_size=(3,3), padding='same',strides=(1,1),
                       activation = 'softplus')(g)
    
    out_image = Concatenate()([out_image,out_uncer])
    
    model = Model(in_image, out_image)
    return model
    

    
###############################################################################


def set_trainable(model, value = True):
	for layer in model.layers:
		layer.trainable = value
	pass


class GradientPenalty(Layer):
    def __init__(self, penalty):
        self.penalty = penalty
        super(GradientPenalty, self).__init__()
        
    def call(self, inputs):
        (y_pred,averaged_samples) = inputs
        gradients = K.gradients(y_pred, averaged_samples)[0]
        norm_grad = K.sqrt(K.sum(K.square(gradients), axis=[1,2,3]))
        gradient_penalty = self.penalty * K.square(K.clip(norm_grad-1.,0.,1e7))
        return (tf.expand_dims(gradient_penalty,-1))

class DriftPenalty(Layer):
    def __init__(self):
        super(DriftPenalty, self).__init__()
        
    def call(self, inputs):
        f_real, f_fake = inputs
        return 0.0005*(K.square(f_real)+K.square(f_fake))


def myAverage(x):
    return (K.mean(x, axis = (1,2)))

    
def Scale(x):
    return(x/1.)


 

    

#%%

# patchgan scale depends on the depth of critic
# best setting: depth of 4 for input size 128
#               depth of 5 for input size 256


#actually, this is not a patchGAN since the receptive field is 382!!!
#change this!!!!!!!! after submission
cNormalization = LayerNormalization
def define_discriminator_gp(in_image,f,norm,k = 4,alpha=0.2,out_act='linear'):
    f=int(f); norm=int(norm)
    
    d = Conv2DEQ(filters=f,kernel_size=(k,k),padding='same',strides=2)(in_image)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DEQ(filters=2*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
    
    d = Conv2DEQ(filters=4*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DEQ(filters=8*f,kernel_size=(k,k),padding='same',strides=2)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    d = Conv2DEQ(filters=8*f,kernel_size=(k,k),padding='same',strides=1)(d)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    
    # d = Conv2DEQ(filters=f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(Identity(scale=False)(d))
    # d = Conv2DEQ(filters=f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    # d = Conv2DEQ(filters=2*f,kernel_size=(k,k),padding='same',strides=2)(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=2*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=2*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    # d = Conv2DEQ(filters=4*f,kernel_size=(k,k),padding='same',strides=2)(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=4*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=4*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    
    # d = Conv2DEQ(filters=8*f,kernel_size=(k,k),padding='same',strides=2)(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=8*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
    # d = Conv2DEQ(filters=8*f,kernel_size=(k,k),padding='same')(d)
    # d = LeakyReLU(.2)(cNormalization(scale=False)(d))
   
    d = Conv2DEQ(filters=1, kernel_size=(k,k), padding='valid')(d)
    patch_out = Activation(out_act)(d)
    # d = GlobalAvgPool2D()(d)
    # patch_out = DenseEQ(units=1, activation='linear')(d)
    model = Model(in_image, patch_out)
   
    return model

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

    
#%%

# define transformer

def define_transformer(inp_dim,out_dim,f, norm,out_act = 'tanh'):
    f=int(f);norm=int(norm)
    # taken from Kwak 2020

    in_image = Input(shape=inp_dim)
    x = Conv2D(32,(3,3),padding='same')(in_image)
    x = BatchNormalization()(x)
    x00 = x
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x00,x])
    x01 = x
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x10 = x
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x10,x])
    x11 = x
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x20 = x
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x20,x])
    x21 = x
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x30 = x
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x30,x])
    x31 = x
    x = MaxPooling2D((2,2))(x)
    
    for dummy in range(5):
        res_in = x
        x = Conv2D(512,(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([res_in,x])
        x = Activation('relu')(x)
        
    x = UpSampling2D((2,2), interpolation = 'nearest')(x)
    x = Concatenate()([x31,x])
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2,2), interpolation = 'nearest')(x)
    x = Concatenate()([x21,x])
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2,2), interpolation = 'nearest')(x)
    x = Concatenate()([x11,x])
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2,2), interpolation = 'nearest')(x)
    x = Concatenate()([x01,x])
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(out_dim[-1],(3,3),padding='same', activation = out_act)(x)
    
    model = Model(inputs = in_image, outputs = x)
    return(model)
        
    
    
