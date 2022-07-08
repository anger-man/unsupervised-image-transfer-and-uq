#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:00:05 2022

@author: c
"""

import os
import configparser

# config = configparser.ConfigParser(allow_no_value = True)

# config.add_section('generator')
# config.set('generator', '; architecture: [unet]')
# config.set('generator', 'arch_g', 'unet')
# config.set('generator', '; learning rate')
# config.set('generator', 'lrg', '5e-5')
# config.set('generator', '; amount of filters for the first convolutional layer')
# config.set('generator', 'fg', '16')
# config.set('generator', '; normalization after convolution: [none,layer,instance]')
# config.set('generator', 'normg', 'instance')

# config.add_section('critic')
# config.set('critic', '; architecture: [dcgan,patchgan]')
# config.set('critic', 'arch_c', 'dcgan')
# config.set('critic', '; learning rate')
# config.set('critic', 'lrc', '2e-5')
# config.set('critic', '; amount of filters for the first convolutional layer')
# config.set('critic', 'fc', '32')
# config.set('critic', '; normalization after convolution: [none,layer,instance]')
# config.set('critic', 'normc', 'none')
# config.set('critic', '; number of intermediate critic iterations (doubled during the first epoch)')
# config.set('critic', 'nc', '15')
# config.set('critic', '; influence of the gradient penalty')
# config.set('critic', 'p', '10')

# config.add_section('training')
# config.set('training', '; epochs')
# config.set('training', 'ep', '100')
# config.set('training', '; batches per epoch (>70)')
# config.set('training', 'bat_per_epo', '80')
# config.set('training', '; minibatch size')
# config.set('training', 'batch_size', '4')
# config.set('training', '; influence patch invariance')
# config.set('training', 'lam', '1')
# config.set('training', '; whether to use GPU acceleration (1) or not (0)')
# config.set('training', 'is_gpu', '1')
# config.set('training',
#             '; whether to use the full-size images (stage=1) or to reduce the spatial dimension by factor 2 (stage=0).')
# config.set('training', 'stage', '1')


# with open('config_file.ini', 'w') as configfile:
#     config.write(configfile)

    

#%%
config = configparser.ConfigParser(allow_no_value = True)
config.read('config_file.ini')
is_gpu = config['training']['is_gpu']

if is_gpu == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    
import sys
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, LayerNormalization
from tensorflow.keras.layers import LeakyReLU, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow_addons.layers import InstanceNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optparse
from generator_networks import define_unet
from critic_networks import define_dcgan_critic, define_patchgan_critic
from critic_networks import Conv2DHE, Identity
from functions import set_trainable, load_data, wasserstein, identity
from functions import DriftPenalty, GradientPenalty, scale_data, update_image_pool
from functions import generate_real_samples, generate_fake_samples
from functions import make_fid, evall_2, plot_curves_gp, validate


    
#%%

parser = optparse.OptionParser()
parser.add_option('--direc', action="store", dest="task", default="IXI/")
options,args = parser.parse_args()

epochs = float(config['training']['ep'])
batch_size = int(config['training']['batch_size'])
bat_per_epo= np.max([float(config['training']['bat_per_epo']),70])
lam = float(config['training']['lam'])

arch_g = config['generator']['arch_g']
fg = float(config['generator']['fg'])
lrg = float(config['generator']['lrg'])
normg = config['generator']['normg']

arch_c = config['critic']['arch_c']
fc = float(config['critic']['fc'])
lrc = float(config['critic']['lrc'])
normc = config['critic']['normc']
nc = np.int(float(config['critic']['nc']))
p = float(config['critic']['p'])
stage = int(config['training']['stage'])


dr = False
paired = True

#%%

task = options.task; os.chdir(task)
files = os.listdir()
if 'results.csv' not in files:
    TABLE = pd.DataFrame(
        columns=['critic','generator','f_critic','f_generator',
             'lr_critic','lr_generator','norm_critic','norm_generator',
             'reconstuction','penalty','num_cri','batches_per_epoch','epochs','SSIM','PSNR','comments'])
    TABLE.to_csv('results.csv',index=False)
    
table = pd.DataFrame(np.array([arch_c,arch_g,fc,fg,lrc,lrg,normc,normg,lam,
                    p,nc,bat_per_epo,epochs,
                    1e-7,1e-7,'']).reshape(1,16),
    columns=['critic','generator','f_critic','f_generator',
         'lr_critic','lr_generator','norm_critic','norm_generator',
         'reconstuction','penalty','num_cri','batches_per_epoch','epochs','SSIM','PSNR','comments'])

TABLE = pd.read_csv('results.csv',encoding = 'unicode_escape')
name = '%04d' %(TABLE.shape[0])
TABLE=TABLE.append(table,ignore_index=True)
TABLE.to_csv('results.csv',index=False)

try:
    os.mkdir('generator_weights')
    os.mkdir('metrics')
    os.mkdir('plots')
    os.mkdir('generator_loss')
    os.mkdir('critic_loss')
except:
    'directories already generated'
    
#%%
    
inp= np.random.permutation([os.path.join('input',f) for f in os.listdir('input')])
tar= np.random.permutation([os.path.join('target',f) for f in os.listdir('target')])
tmp = [os.path.join('evaluation', f) for f in os.listdir('evaluation')]
df_e = pd.DataFrame(columns=['input','target'])
df_e['input']= pd.Series(np.sort(np.array(tmp)[np.array(['input' in f for f in tmp])]))
df_e['target']= pd.Series(np.sort(np.array(tmp)[np.array(['target' in f for f in tmp])]))
evaluation = df_e


inp_dim=load_data(task,1,0, inp, tar,domain='input').shape[1:]
out_dim=load_data(task,1,0, inp, tar,domain='target').shape[1:]
stage_dims = np.array([np.array(inp_dim)[:2]/2**t for t in np.arange(1,-1,-1)],dtype=int)
print(inp_dim); print(out_dim)
print(stage_dims)
    
#%%

class PatchExtractor(Layer):
    def __init__(self, coord):
        self.coord = coord
        super(PatchExtractor,self).__init__()
        
    def call(self,inputs):
        ii = tf.constant(0)
        inputs = tf.cast(inputs,tf.float32)
        res = inputs[0:1]
        coord = self.coord
        
        c = lambda ii,res,coord: ii<tf.shape(coord)
        def b(ii,res,coord):
            pwidth = coord[ii]
            i1 = coord[ii+1]; i2=coord[ii+2]
            j = tf.cast(ii/3,tf.int32)
            tmp = inputs[j:j+1,i1:i1+pwidth,i2:i2+pwidth,:]
            tmp = tf.image.resize(tmp,[tf.shape(inputs)[1],tf.shape(inputs)[2]],method='bicubic')
            res = tf.concat([res,tmp],axis=0)
            return ii+3,res,coord
        
        ii,res,coord = tf.while_loop(c,b,[ii,res,coord],shape_invariants=
                                     [ii.get_shape(),tf.TensorShape([None,None,None,None]),
                                      tf.TensorShape([None])])
        return(res[1:])
    
    
coord_real=[]
for dummy in range(batch_size):
    pwidth=(np.random.uniform(.7,1.)*stage_dims[stage][0]).astype(np.int32)
    coord_real.append(pwidth)
    coord_real.append(np.random.choice(range(stage_dims[stage][0]-pwidth)))
    coord_real.append(np.random.choice(range(stage_dims[stage][0]-pwidth)))
coord = K.variable(coord_real,tf.int32)
trans = K.variable(0.)

#%%

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(gCtoD, dD1, dD2, inp_dim,lr, coord,alpha,beta,lam):
    set_trainable(gCtoD, True)
    set_trainable(dD1, False)
    set_trainable(dD2, False)
    
    def l1(l):
        x0 = l[0]; x1 = K.clip(l[1],-1.,1.);
        uncer = (l[2]+1e-6)
        return tf.expand_dims(K.mean(K.abs(x0-x1)*np.sqrt(2)/uncer+K.log(1.+np.sqrt(2)*uncer),
                                     axis=(1,2,3)),-1)
    #aktuell uncer map from lpf(patch) image
    #second cycle with uncer map patch from ori image ?
    
  
    input_ori = Input(shape=inp_dim)
    input_lpf = PatchExtractor(coord)(input_ori)
    gen_ori = gCtoD(input_ori)
    gen_lpf = gCtoD(input_lpf)
    outimg_ori = gen_ori[...,:1]
    outimg_lpf = gen_lpf[...,:1]
    uncerori   = gen_ori[...,1:]
    uncerlpf   = gen_lpf[...,1:]
    
    
    output_ori = dD1(outimg_ori)
    output_lpf = dD2(outimg_lpf)
    
    circ2 = Lambda(l1)([outimg_lpf, 
                        PatchExtractor(coord)(outimg_ori),
                        PatchExtractor(coord)(uncerori)])
    circ1 = Lambda(l1)([outimg_lpf, 
                        PatchExtractor(coord)(outimg_ori),
                        uncerlpf])

    
    model = Model([input_ori], [output_ori, output_lpf, circ1, circ2])
    # model = Model(input_gen,output_d)
    model.compile(loss=[wasserstein,wasserstein, identity, identity],
            loss_weights=[alpha,beta,lam,lam], 
            optimizer=tf.keras.optimizers.Adam(beta_1 = 0., beta_2 = 0.9, lr = lr))
    return model




def define_critic_with_gp(dD1,inp_dim,penalty,lrc):

    set_trainable(dD1, True)

    inp_real = Input(shape=inp_dim)
    cri_real = dD1(inp_real)
    
    inp_fake = Input(shape=inp_dim)
    cri_fake = dD1(inp_fake)
    
    
    inp_mix  = Input(shape=inp_dim)
    temp = dD1(inp_mix)
    gp   = GradientPenalty(penalty)([temp,inp_mix])
    pen  = DriftPenalty()([cri_real, cri_fake])
    
    model = Model([inp_real,inp_fake,inp_mix], [cri_real,cri_fake,pen,gp])

    model.compile(loss=[wasserstein,wasserstein,identity,identity],
            loss_weights=[1,1,1,1],
            optimizer=tf.keras.optimizers.Adam(beta_1 = 0., beta_2 = 0.9, lr=lrc))
    return model

#%%

def add_disblock(dis,f,stage,length,back,channel,norm = normc):
    
    if norm=='none':
        cNormalization = Identity
    elif norm=='layer':
        cNormalization = LayerNormalization
    elif norm=='instance':
        cNormalization = InstanceNormalization
    else:
        print('No valid norm defined'); pass;
        
    in_image0 = Input(shape=[*stage_dims[stage],channel])
    d = Conv2DHE(filters=np.min([512,f]),kernel_size=(4,4),padding='same',strides=2)(in_image0)
    d = LeakyReLU(.2)(cNormalization(scale=False)(d))
 
    for lay in np.arange(length-back,length,1):
        d = dis.layers[lay](d)
    model0 = Model(in_image0, d)
    return(model0)


dD_list = list()
dD_transitions = list()
in_image = Input(shape=out_dim)
if arch_c == 'dcgan':
    disD = define_dcgan_critic(in_image, fc, norm = normc)
else:
    disD = define_patchgan_critic(in_image, fc, norm = normc)
lenD=len(disD.layers)
dD_list.append(add_disblock(disD,2* fc,0,lenD,13,out_dim[-1]))
dD_list.append(disD)


if arch_c == 'dcgan':
    disDscale = define_dcgan_critic(in_image, fc, norm = normc)
else:
    disDscale = define_patchgan_critic(in_image, fc, norm = normc)
lenDscale=len(disDscale.layers)
dD_list_scale = list()
dD_list_scale.append(add_disblock(disDscale,2* fc,0,lenD,13,out_dim[-1]))
dD_list_scale.append(disDscale)

#%%

def add_genblock(gen,f,stage,back,forw,inp_dim,out_dim, norm=normg):
    
    if norm=='none':
        gNormalization = Identity
    elif norm=='layer':
        gNormalization = LayerNormalization
    elif norm=='instance':
        gNormalization = InstanceNormalization
    else:
        print('No valid norm defined'); pass;
        
    
    in_image0 = Input(shape=[*stage_dims[stage],inp_dim[-1]])
    g = Conv2DHE(filters=f//2,kernel_size=(3,3),padding='same',dropout=dr)(in_image0)
    g = LeakyReLU(.0)(gNormalization()(g))
    tensors = list()
    for lay in np.arange(32-back,56+forw-10,1):
        if gen.layers[lay].__class__.__name__ == 'Concatenate':
            i=5
            while(True):
                if tensors[len(tensors)-i].shape[1] == g.shape[1]:
                    g = gen.layers[lay]([tensors[len(tensors)-i],g])
                    break;
                else:
                    i+=1
        else:
            g = gen.layers[lay](g)
        tensors.append(g)
    foo = g
    
    g = Conv2DHE(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,tensors[5]])
    # g = Activation('linear')(g)
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_img = Conv2DHE(filters=out_dim[-1],kernel_size=(3,3),padding='same',activation='tanh')(g)  
    
    g = Conv2DHE(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,tensors[5]])
    # g = Activation('linear')(g)
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DHE(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_uncer = Conv2DHE(filters=1,kernel_size=(3,3),padding='same',activation='softplus')(g)
    
    out_img = Concatenate()([out_img, out_uncer])
    model0 = Model(in_image0, out_img)
    return(model0)


if arch_g == 'unet':
    genCtoD = define_unet(inp_dim,out_dim,f=fg,dr=dr,norm=normg)
else:
    print('invalid architecture')
    sys.exit(1)
lenCtoD = len(genCtoD.layers)
gCtoD_list = list();


gCtoD_list.append(
    add_genblock(genCtoD,2*fg,stage=0,back=21,forw=33,inp_dim=inp_dim, out_dim = out_dim))
gCtoD_list.append(genCtoD)


#%%

lossD=[]; genlossD=[]; dDval=[]; DIF1=[]; DIF2=[]; gradD = []; gradC = []

if paired==False:
    try:
        inc_model = InceptionV3(include_top=False, pooling='avg',weights=None,
                                                input_shape=[299,299,3])
        inc_model.set_weights(np.load('inc_weights.npy',allow_pickle=True))
    except:
        inc_model = InceptionV3(include_top=False, pooling='avg',weights='imagenet',
                                            input_shape=[299,299,3])



i=-1
runs = 0
steps = bat_per_epo * epochs


for STAGE in [stage]:
    K.set_value(trans,0.)
    inp= np.random.permutation(inp)
    tar= np.random.permutation(tar)

    gCtoD = gCtoD_list[STAGE]
    dD1 = dD_list[STAGE]

    alpha = K.variable(.5)
    beta  = K.variable(.5)
    if lam == 0:
        K.set_value(alpha,1)
        K.set_value(beta,0)
    print('alpha: ',K.get_value(alpha),'\nbeta: ',K.get_value(beta))
    varlam = K.variable(0.)
    
    critic_D1 = define_critic_with_gp(dD1,[*stage_dims[STAGE],out_dim[-1]],p,lrc)
    function_CtoD=define_composite_model(gCtoD,dD1,dD1,[*stage_dims[STAGE],inp_dim[-1]], lrg,coord,
                                         alpha,beta,lam=varlam)
   
    patches = disD.output_shape[1:]
    placeholder_img = np.zeros((batch_size,1))
    try:
        placeholder = np.zeros((batch_size,patches[0],patches[1],1))
    except:
        placeholder = placeholder_img
    
    count=0; foo = 0
    poolD = list(); poolDlpf = list()
    runs += 1
    
    print(dD1.summary(), flush=True)
    print(gCtoD.summary(), flush=True)
    
#%%
    while(i < steps):
        i+=1
        
        NC = nc if i>bat_per_epo else 2*nc
        print(NC)
        for iters in range(NC):
            x_realC, y_real = generate_real_samples(load_data(task,batch_size,count,
                                                            inp,tar,domain='input'), patches)
            x_realC = scale_data(x_realC, stage_dims[STAGE])            
            x_realDori, y_real = generate_real_samples(load_data(task,batch_size,count,
                                                            inp,tar,domain='target'), patches)
            x_realDori = scale_data(x_realDori, stage_dims[STAGE])
            x_realDlpf = PatchExtractor(coord)(x_realDori)
            x_fakeDori, y_fake = generate_fake_samples(gCtoD, x_realC, patches, out_dim[-1])
            x_fakeDori = update_image_pool(poolD, x_fakeDori)
            x_mixDori = np.zeros(x_realDori.shape);
            for b in range(x_mixDori.shape[0]):
                eps = np.random.uniform()
                x_mixDori[b] = eps*x_realDori[b] + (1.-eps)*x_fakeDori[b];
            count += batch_size
            ###################################################################
            x_fakeDlpf, y_fake = generate_fake_samples(gCtoD, PatchExtractor(coord)(x_realC), patches, out_dim[-1])
            x_fakeDlpf = update_image_pool(poolDlpf, x_fakeDlpf)
            x_mixDlpf = np.zeros(x_realDlpf.shape);
            for b in range(x_mixDlpf.shape[0]):
                eps = np.random.uniform()
                x_mixDlpf[b] = eps*x_realDlpf[b] + (1.-eps)*x_fakeDlpf[b];
            ###################################################################
                
            oriLoss = critic_D1.train_on_batch([x_realDori,x_fakeDori,x_mixDori], 
                                               [y_real,y_fake,placeholder, placeholder_img])
            lpfLoss = critic_D1.train_on_batch([x_realDlpf,x_fakeDlpf,x_mixDlpf], 
                                               [y_real,y_fake,placeholder, placeholder_img])
            w_ori = -(oriLoss[1]+oriLoss[2]); w_lpf=w_ori
            if lam > 0:
                w_lpf = -(lpfLoss[1]+lpfLoss[2])
                                                                          
                                                                    
        
        print(str('W1_Dori %08.3f, PENori %+08.3f, gradLori %+08.3f' 
                  %(-(oriLoss[1]+oriLoss[2]),oriLoss[3],oriLoss[4])),flush=True)
        print(str('W1_Dlpf %08.3f, PENlpf %+08.3f, gradLlpf %+08.3f' 
                  %(-(lpfLoss[1]+lpfLoss[2]),lpfLoss[3],lpfLoss[4])),flush=True)
        
        
        
        
        
        coord_real=[]
        for dummy in range(batch_size):
            pwidth=(np.random.uniform(.7,1.)*stage_dims[stage][0]).astype(np.int32)
            coord_real.append(pwidth)
            coord_real.append(np.random.choice(range(stage_dims[stage][0]-pwidth)))
            coord_real.append(np.random.choice(range(stage_dims[stage][0]-pwidth)))
        K.set_value(coord,coord_real)
        K.set_value(varlam, np.min([lam, lam*i/bat_per_epo])); print(K.get_value(varlam))
        
        if i%10==0:
            w1 = -oriLoss[1]; w2 = oriLoss[2]
            print(str('criticD r %08.3f f %08.3f' %(w1,w2)),flush=True)
            w1 = -lpfLoss[1]; w2 = lpfLoss[2]
            print(str('criticD r %08.3f f %08.3f' %(w1,w2)),flush=True)
            
        if i%10==0:
            vali = validate(task,dD1,dD1,gCtoD,evaluation,stage_dims[STAGE],PatchExtractor,coord)
            for dummy in range(10):
                dDval.append(vali)
        
    
        K.set_value(trans, np.min([1., i/steps])); 
        x_realC, y_real = generate_real_samples(load_data(task,batch_size,count,
                                                        inp,tar,domain='input'), patches)
        x_realC = scale_data(x_realC, stage_dims[STAGE]);
        
     
        genlossD.append(function_CtoD.train_on_batch([x_realC],[y_real,y_real,placeholder_img,placeholder_img]))
                                               
        print('count: %06d; genDori: %08.3f; genDlpf: %08.3f; circ1: %08.3f; circ2: %08.3f; progress: %04.1f pp.\n ' 
              %(count,*genlossD[-1][1:],trans*100))
        lossD.append(.5*(w_ori+w_lpf))
        
        
    
        if (i+1) % bat_per_epo == 0:
            tmp = dD1.get_weights()
            tmp = [tmp[k] for k in range(len(tmp)) if type(tmp[k])==np.ndarray]
            criticW=np.hstack([item.reshape(-1) for sublist in tmp for item in sublist])
            plot_curves_gp(lossD,genlossD,lam,dDval,criticW,name+'uapi')
            dif1,dif2 = evall_2(i, gCtoD, evaluation,name+'uapi',task,stage_dims[STAGE],paired=paired)
            
            if paired==False:
                dif1,dif2 = make_fid(task,evaluation,stage_dims,STAGE,count,patches,out_dim,gCtoD,inc_model)
                DIF1.append(dif1); DIF2.append(dif2)
                plt.figure()
                plt.plot(DIF1); plt.plot(DIF2)
                plt.title(str('%.3f,  %.3f' %(np.min(DIF1),np.min(DIF2))))
                plt.savefig( 'metric/%s.pdf' %(name+''),dpi=300)
            
            else:
                DIF1.append(dif1); DIF2.append(dif2)
                plt.figure()
                plt.plot(DIF1); plt.plot(DIF2)
                plt.title(str('%.3f,  %.3f' %(np.max(DIF1),np.max(DIF2))))
                plt.savefig( 'metric/%s.pdf' %(name+''),dpi=300)
        
            wg1 = gCtoD.get_weights()
            np.save(str('generator_weights/%s_%04d' % (name,i+1)),wg1, allow_pickle=True)
            
            inp= np.random.permutation(inp)
            tar= np.random.permutation(tar)
            
     
#%%
TABLE=pd.read_csv('results.csv',encoding = 'unicode_escape')
if paired:
    TABLE.FID1[int(name)]=np.max(DIF1)
    TABLE.FID2[int(name)]=np.max(DIF2)
else:
    TABLE.FID1[int(name)]=np.min(DIF1)
    TABLE.FID2[int(name)]=np.min(DIF2)
    
TABLE.to_csv('TABLEfinal.csv',index=False)



