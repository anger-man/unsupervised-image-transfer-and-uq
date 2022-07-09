#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:58:59 2022

@author: c
"""

#%%
import numpy as np
from matplotlib.pyplot import imread
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, AveragePooling2D
import skimage
import scipy
import matplotlib.pyplot as plt
import skimage.measure
from skimage.metrics import structural_similarity as ssim

#%%

# Metrics
def psnr(y_true,y_pred):
    res = 20*np.log10(np.max(y_true))-10*np.log10(np.mean(np.square(y_true-y_pred)))
    return res


#%%

# Tensorflow functions

def set_trainable(model, value = True):
	for layer in model.layers:
		layer.trainable = value
	pass


def wasserstein(y_true,y_pred):
    return(K.mean(y_true*y_pred))


def identity(y_true,y_pred):
    return(y_pred-y_true)


class DriftPenalty(Layer):
    def __init__(self):
        super(DriftPenalty, self).__init__()
        
    def call(self, inputs):
        f_real, f_fake = inputs
        return 0.0005*(K.square(f_real)+K.square(f_fake))


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
#%%

# data functions

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


def generate_fake_samples(generator, dataset, patch_shape, out_channel):
    # generate fake instance
    X = generator.predict(dataset,batch_size=2)[...,:out_channel]
    # X = np.concatenate((X[0],X[1],X[2]),axis=0)
    # create 'fake' class labels (0)
    try:
        y = np.ones((len(X), patch_shape[0], patch_shape[1], 2))
    except:
        y = np.ones((len(X), 1))
    return X, y


def generate_real_samples(dataset, patch_shape):
    X = dataset
    # generate 'real' class labels (1)
    try:
        y = -np.ones((len(X), patch_shape[0], patch_shape[1], 2))
    except:
        y = -np.ones((len(X), 1))
    return X, y
#%% 

# plot functions

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
        avg_pool = AveragePooling2D(pool_size=(2,2),padding='valid',strides=(1,1))
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


def plot_curves_gp(dB,gB,alpha,dDval,criticW,name):
    dB=np.array(dB); gB = np.array(gB)
    dDval = np.array(dDval)
    
    plt.figure()
    l2 = np.convolve(dB[50:],np.ones(20)/20,mode='valid')
    b,=plt.plot(l2); 
    l3 = np.array(dDval[:,0])
    l4 = np.array(dDval[:,1])
    c,=plt.plot(np.convolve(l3[50:],np.ones(20)/20,mode='valid'),linewidth=.3);
    d,=plt.plot(np.convolve(l4[50:],np.ones(20)/20,mode='valid'),linewidth=.3); 
    plt.legend((b,c,d),('W1-distance train','W1-distance test full-size','W1-distance test patches'))
    plt.title(str('W1_D: %.2f' %(np.mean(l2[max(len(l2)-100,0):]))))
    plt.savefig('critic_loss/%s.pdf' %name,dpi=200)
    
    plt.figure(figsize=(8,4)) 
    l2 = np.mean(gB[50:,1:3],axis=-1)
    b,=plt.plot(np.convolve(l2,np.ones(20)/20,mode='valid')); 
    l3 = alpha*np.sum(gB[50:,3:], axis=-1)
    c,=plt.plot(np.convolve(l3,np.ones(20)/20,mode='valid')); 
    plt.legend((b,c),('adversarial','patch invariance'))
    plt.savefig('generator_loss/%s.pdf' %name,dpi=200)
