#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:35:42 2022

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
from tensorflow.keras.layers import  Conv2D, LeakyReLU,LayerNormalization
from tensorflow.keras.layers import Activation, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Concatenate, Input
from tensorflow_addons.layers import InstanceNormalization
from critic_networks import Identity, Conv2DHE
import numpy as np
#%%

"""
Building blocks for the standard U-Net implementation proposed by 
Ronneberger et al in 2015."""

def unet_enc(f, input_layer, red=True, gnorm=Identity, dropout=False):
    f = np.min([f,512])
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dropout)(input_layer)
    g = LeakyReLU(.0)(gnorm()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dropout)(g)
    g = LeakyReLU(.0)(gnorm()(g))
    shortcut = g
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dropout,strides=2)(g)
    g = LeakyReLU(.0)(gnorm()(g))
    return(g,shortcut)

def unet_dec(f, input_layer, skip, short=True, gnorm = Identity, dropout=False):
    g = UpSampling2D((2,2),interpolation='nearest')(input_layer)
    g = Conv2DHE(filters=f, kernel_size=(4,4),padding='same',dropout=dropout)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Concatenate()([g,skip])
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dropout)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dropout)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    return(g)


#%%

def define_unet(inp_dim,out_dim,f, norm = 'none', out_act = 'tanh',dr=False):
    
    """ A standard U-net with 5 downsampling steps. The decoder is split into
    two identical output branches after the last upsampling operation in order
    to predict the translated image as well as the corresponding uncertainty map."""
    
    f=int(f);
    
    if norm=='none':
        gnorm = Identity
    elif norm=='layer':
        gnorm = LayerNormalization
    elif norm=='instance':
        gnorm = InstanceNormalization
    else:
        print('No valid norm defined'); pass;
    
    in_image = Input(shape=inp_dim)
    g = in_image
    g = Conv2DHE(filters=f//2,kernel_size=(3,3), padding = 'same',dropout=dr)(g)
    g = LeakyReLU(.0)(gnorm()(g))
    g, g_0 = unet_enc(f,    g, red= True, gnorm=gnorm,dropout=dr)
    g, g_1 = unet_enc(2*f,  g, red= True, gnorm=gnorm,dropout=dr)
    g, g_2 = unet_enc(4*f,  g, red= True, gnorm=gnorm,dropout=dr)
    g, g_3 = unet_enc(8*f,  g, red= True, gnorm=gnorm,dropout=dr)
    g, g_4 = unet_enc(16*f, g, red= True, gnorm=gnorm,dropout=dr)
    foo, g = unet_enc(32*f, g, red= True, gnorm=gnorm,dropout=dr)
    g = unet_dec(16*f,g, g_4, gnorm=gnorm,dropout=dr)
    g = unet_dec(8*f, g, g_3, gnorm=gnorm,dropout=dr)
    g = unet_dec(4*f, g, g_2, gnorm=gnorm,dropout=dr)
    g = unet_dec(2*f, g, g_1, gnorm=gnorm,dropout=dr)
    foo = UpSampling2D((2,2),interpolation='nearest')(g)
    
    g = Conv2DHE(filters=f, kernel_size=(4,4),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Concatenate()([g,g_0])
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    out_image = Conv2DHE(filters=out_dim[-1], kernel_size=(3,3), padding='same',strides=(1,1),
                       activation = out_act)(g)
    
    g = Conv2DHE(filters=f, kernel_size=(4,4),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Concatenate()([g,g_0])
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.1)(gnorm()(g))
    out_uncer = Conv2DHE(filters=1, kernel_size=(3,3), padding='same',strides=(1,1),
                       activation = 'softplus')(g)
    
    out_image = Concatenate()([out_image,out_uncer])
    
    model = Model(in_image, out_image)
    return model


#%%