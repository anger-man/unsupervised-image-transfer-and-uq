
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:11:48 2022

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
import time
logging.disable(logging.WARNING)
from tensorflow.keras.optimizers import Adam #adadelta heavily fails for all tasks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Layer, AveragePooling2D, Concatenate
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optparse
parser = optparse.OptionParser()
from source_onesided import scale_data, set_trainable, GradientPenalty, load_data,validate
from source_onesided import gNormalization, Conv2DEQ, LeakyReLU, UpSampling2D
from source_onesided import define_discriminator_gp, define_unet, generate_real_samples,make_fid
from source_onesided import generate_fake_samples, plot_curves_gp, evall_2
from source_onesided import DriftPenalty,update_image_pool,Identity
from tensorflow.keras.applications.inception_v3 import InceptionV3
###############################################################################


def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
    
#W = np.load('resnet50_weights.npy',allow_pickle=True)

parser.add_option('-g', '--lrg', action="store", dest="lrg", type='float', default=5e-5)
parser.add_option('--p', action="store", dest="p", type='float', default=10)
parser.add_option('-c', '--lrc', action="store", dest="lrc", type='float', default=2e-5)
parser.add_option('-n', '--nc', action="store", dest="nc", type='int', default=15)
parser.add_option('-d', '--dr', action="store", dest="dr", type='int', default=0)
parser.add_option('-l', '--lam', action="store", dest="lam", type='float', default=.1)
parser.add_option('--task', action="store", dest="task", default="hea/")
parser.add_option('--wait', action="store", dest="N", type='int', default=0)
parser.add_option('--fg', action="store", dest="fg", type='int', default=16)
parser.add_option('--fc', action="store", dest="fc", type='int', default=32)
parser.add_option('--steps', action="store", dest="steps", type='int', default=100)
parser.add_option('--stage', action='store',dest = 'stage',type='int',default=1)

options,args = parser.parse_args()
time.sleep(options.N)
task = options.task; 
os.chdir(task)

files = os.listdir()
if 'TABLEfinal.csv' not in files:
    TABLE = pd.DataFrame(
        columns=['architecture','dim','fg','fc',
                 'lr_genC','lr_genD','lr_criC','lr_criD',
                 'penalty','num_cri','cyc_loss','drop',
                 'steps','FID1','FID2',
                 'comments'])
    TABLE.to_csv('TABLEfinal.csv',index=False)
    


if task in ['tex/','ixi/','rir/','kne/','dt4/']:
    paired=True;
else:
    paired = False
if task in ['ixi/','rir/','kne/','hea/']:
    is_rgb = False;
else:
    is_rgb=True



#%%

def mymae(l):
    ytrue = l[0]; ypred = l[1]
    return(K.mean(K.abs(ytrue-ypred)))

def mymse(y_true,y_pred):
    return(K.mean(K.square(y_true-y_pred)))

def wasserstein(y_true,y_pred):
    return(K.mean(y_true*y_pred))

def identity(y_true,y_pred):
    return(y_pred-y_true)



class MyFilter(Layer):
    def __init__(self, coord):
        self.coord = coord
        super(MyFilter,self).__init__()
        
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


    
#%%

nc=options.nc; dr = options.dr; lam = options.lam
dc=0.; dd=0.; p=options.p; cycl = mymae; 
lrgc = options.lrg; lrgd = options.lrg; lrcc = options.lrc; lrcd = options.lrc

df = pd.read_csv('filenames_%s.csv' %task[:3])
inp= np.random.permutation(df['input'].dropna())
tar= np.random.permutation(df['target'].dropna())
evaluation = pd.read_csv('evaluation_%s.csv' %task[:3])
inp_dim=load_data(task,1,0, inp, tar,domain='input').shape[1:]
out_dim=load_data(task,1,0, inp, tar,domain='target').shape[1:]
stage_dims = np.array([np.array(inp_dim)[:2]/2**t for t in np.arange(1,-1,-1)],dtype=int)
print(inp_dim); print(out_dim)
print(stage_dims)


table = pd.DataFrame(np.array(['unet',stage_dims[options.stage][0],options.fg,options.fc,lrgc,lrgd,lrcc,lrcd,
                    p,nc,str(lam),dr,options.steps,
                    9e5,9e5,'']).reshape(1,16),
    columns=['architecture','dim','fg','fc','lr_genC','lr_genD','lr_criC','lr_criD',
                            'penalty','num_cri',
                            'cyc_loss','drop','steps',
                            'FID1','FID2','comments'])

TABLE = pd.read_csv('TABLEfinal.csv',encoding = 'unicode_escape')
name = '%04d' %(TABLE.shape[0])

TABLE=TABLE.append(table,ignore_index=True, sort=True)
TABLE.to_csv('TABLEfinal.csv',index=False)

    

batch_size = np.max([2,2**(4-options.stage)])
coord_real=[]
for dummy in range(batch_size):
    pwidth=(np.random.uniform(.7,1.)*stage_dims[options.stage][0]).astype(np.int32)
    coord_real.append(pwidth)
    coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
    coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
coord = K.variable(coord_real,tf.int32)
trans = K.variable(0.)


        

#%%

# x=load_data(1,0,inp,tar,'target')[0]
# plt.imshow(x);plt.show()

# mat = np.ones(x.shape)
# h,w = mat.shape[:2]
# mat[h//2-4:h//2+4,w//2-4:w//2+4]=0.1
# tmp = np.fft.fftshift(np.fft.fft2(x,axes=(0,1)))
# tmp = tmp*mat
# tmp = np.fft.ifft2(np.fft.ifftshift(tmp), axes=(0,1))
# plt.imshow(np.real(tmp));plt.show()

# mat = np.ones(x.shape)
# h,w = mat.shape[:2]
# mat[h//2-4:h//2+4,w//2-4:w//2+4]=10
# tmp = np.fft.fftshift(np.fft.fft2(tmp,axes=(0,1)))
# tmp = tmp*mat
# res = np.fft.ifft2(np.fft.ifftshift(tmp), axes=(0,1))
# plt.imshow(np.real(res));plt.show()

# np.max(np.abs(x-np.real(res)))

###################################################################################

# x = load_data(1,0,inp,tar,'input')[0]
# plt.imshow(x*.5+.5); plt.show()

# conversion = np.matrix([[0.299,-0.168935,0.499813],[0.587,-0.331664, -0.418431],
#                        [0.114,0.50059,-0.081282]])
# convinv = np.linalg.inv(conversion)
# ycbcr = np.tensordot(x,conversion,axes=((2,1))) 

# tmp = ycbcr[...,0:1]
# mat = np.ones(tmp.shape)
# h,w = mat.shape[:2]
# mat[h//2-4:h//2+4,w//2-4:w//2+4]=0.1
# tmp = np.fft.fftshift(np.fft.fft2(tmp,axes=(0,1)))
# tmp = tmp*mat
# tmp = np.real(np.fft.ifft2(np.fft.ifftshift(tmp), axes=(0,1)))
# plt.imshow(np.abs(tmp));plt.show()
# ycbcr[...,0] = tmp[...,0]
# rgb = np.tensordot(ycbcr,convinv,axes=((2,1))) 


# plt.imshow(rgb*.5+.5); plt.show()



#%%


    
    
# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(gCtoD, dD1, dD2, inp_dim,lr, coord,alpha,beta,lam):
    set_trainable(gCtoD, True)
    set_trainable(dD1, False)
    set_trainable(dD2, False)
    
    def l1(l):
        x0 = l[0]; x1 = K.clip(l[1],-1.,1.);
        return tf.expand_dims(K.mean(K.abs(x0-x1),axis=(1,2,3)),-1)
    #aktuell uncer map from lpf(patch) image
    #second cycle with uncer map patch from ori image ?
    
  
    input_ori = Input(shape=inp_dim)
    input_lpf = MyFilter(coord)(input_ori)
    gen_ori = gCtoD(input_ori)
    gen_lpf = gCtoD(input_lpf)
    
    outimg_ori = gen_ori[...,:out_dim[-1]]
    outimg_lpf = gen_lpf[...,:out_dim[-1]]
    
    
    output_ori = dD1(outimg_ori)
    output_lpf = dD2(outimg_lpf)

    # circ1 = Lambda(l1)([gen_ori, MyFilter(inv_lpf)(gen_lpf)])
    circ2 = Lambda(l1)([outimg_lpf,MyFilter(coord)(outimg_ori)])
    circ1 = circ2

    
    model = Model([input_ori], [output_ori, output_lpf, circ1, circ2])
    # model = Model(input_gen,output_d)
    model.compile(loss=[wasserstein,wasserstein, identity, identity],
            loss_weights=[alpha,beta,lam,lam], optimizer=Adam(beta_1 = 0., beta_2 = 0.9, lr = lr))
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
                optimizer=Adam(beta_1 = 0., beta_2 = 0.9, lr=lrc))
    return model





#%%

class MyAddition(Layer):
   
    def __init__(self, trans):
       self.trans = trans
       super(MyAddition, self).__init__()
        
    def call(self, l):
        x1,x2 = l
        return ((1.-self.trans)*x1 + self.trans*x2)

def add_disblock(dis,f,stage,length,back,channel):
    in_image0 = Input(shape=[*stage_dims[stage],channel])
    d = Conv2DEQ(filters=np.min([512,f]),kernel_size=(4,4),padding='same',strides=2)(in_image0)
    d = LeakyReLU(.2)(Identity(scale=False)(d))
 
    for lay in np.arange(length-back,length,1):
        d = dis.layers[lay](d)
    model0 = Model(in_image0, d)
    return(model0)

def add_transition(dis, prev_dis):
    in_image0 = Input(dis.input_shape[1:])
    d = dis.layers[1](in_image0)
    for dindex in np.arange(2,7,1):
        d = dis.layers[dindex](d)
    foo = AveragePooling2D(2)(in_image0)
    foo = prev_dis.layers[1](foo)
    foo = prev_dis.layers[2](foo)
    foo = prev_dis.layers[3](foo)
    d = MyAddition(trans)([foo,d])
    for lay in np.arange(7,len(dis.layers),1):
        d = dis.layers[lay](d)
    model0 = Model(in_image0, d)
    return(model0)
###############################################################################
F = options.fc

dD_list = list()
dD_transitions = list()
in_image = Input(shape=out_dim)
disD = define_discriminator_gp(in_image, F, norm = 0)
lenD=len(disD.layers)

dD_list.append(add_disblock(disD,2* F,0,lenD,13,out_dim[-1]))
dD_list.append(disD)



# dD_transitions.append(disD)
# dD_transitions.append(add_transition(dD_list[1],dD_list[0]))
# dD_transitions.append(add_transition(dD_list[2],dD_list[1]))
# dD_transitions.append(add_transition(dD_list[3],dD_list[2]))
# dD_transitions.append(add_transition(dD_list[4],dD_list[3]))

###############################################################################
disDr = define_discriminator_gp(in_image, F, norm = 0)
lenD=len(disDr.layers)
dD_list_rot = list()


dD_list_rot.append(add_disblock(disDr,2* F,0,lenD,13,out_dim[-1]))
dD_list_rot.append(disDr)

#%%

def add_genblock(gen,f,stage,back,forw,inp_dim,out_dim):
    in_image0 = Input(shape=[*stage_dims[stage],inp_dim[-1]])
    g = Conv2DEQ(filters=f//2,kernel_size=(3,3),padding='same',dropout=dr)(in_image0)
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
    
    g = Conv2DEQ(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,tensors[5]])
    # g = Activation('linear')(g)
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_img = Conv2DEQ(filters=out_dim[-1],kernel_size=(3,3),padding='same',activation='tanh')(g)  
    
    g = Conv2DEQ(filters=f, kernel_size=(3,3),padding='same',dropout=dr)(foo)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Concatenate()([g,tensors[5]])
    # g = Activation('linear')(g)
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    g = Conv2DEQ(filters=f, kernel_size=(3,3), padding='same',dropout=dr)(g)
    g = LeakyReLU(.0)(gNormalization()(g))
    out_uncer = Conv2DEQ(filters=1,kernel_size=(3,3),padding='same',activation='softplus')(g)
    
    out_img = Concatenate()([out_img, out_uncer])
    model0 = Model(in_image0, out_img)
    # model0.summary()
    return(model0)

def add_transition(gen, prev_gen):
    in_image0 = Input(gen.input_shape[1:])
    g = gen.layers[1](in_image0)
    tensors = list()
    for dindex in np.arange(2,11,1):
        g = gen.layers[dindex](g)
        tensors.append(g)
        
    foo = AveragePooling2D(2)(in_image0)
    foo = prev_gen.layers[1](foo)
    foo = prev_gen.layers[2](foo)
    foo = prev_gen.layers[3](foo)
    g = MyAddition(trans)([foo,g])
    
    for lay in np.arange(11,len(gen.layers),1):
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
        if lay == len(gen.layers)-13:
            foo = prev_gen.layers[-1](g)
            
            
    foo = UpSampling2D((2,2),interpolation= 'nearest')(foo)
    g = MyAddition(trans)([foo,g])
    
    model0 = Model(in_image0, g)
    # model0.summary()
    return(model0)

###############################################################################
F = options.fg

genCtoD = define_unet(inp_dim,out_dim,f=F,dr=dr)
#genCtoD.summary()
lenCtoD = len(genCtoD.layers)
gCtoD_list = list(); gCtoD_transitions = list()


gCtoD_list.append(
    add_genblock(genCtoD,2*F,stage=0,back=21,forw=33,inp_dim=inp_dim, out_dim = out_dim))
gCtoD_list.append(genCtoD)

# gCtoD_transitions.append(genCtoD)
# gCtoD_transitions.append(add_transition(gCtoD_list[1],gCtoD_list[0]))
# gCtoD_transitions.append(add_transition(gCtoD_list[2],gCtoD_list[1]))
# gCtoD_transitions.append(add_transition(gCtoD_list[3],gCtoD_list[2]))
# gCtoD_transitions.append(add_transition(gCtoD_list[4],gCtoD_list[3]))

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


for STAGE in [options.stage]:
    K.set_value(trans,0.)
    
    
    bat_per_epo = 120
    steps = bat_per_epo * options.steps
    
    inp= np.random.permutation(df['input'].dropna())
    tar= np.random.permutation(df['target'].dropna())
    
    
    if runs%2 == 0:
        
        gCtoD = gCtoD_list[STAGE]
        dD1 = dD_list[STAGE]
        # dD2 = dD_list_rot[STAGE]
        
        
    else:
        gCtoD = gCtoD_transitions[STAGE]
        dD = dD_transitions[STAGE]
        
    alpha = K.variable(.5)
    beta  = K.variable(.5)
    if lam == 0:
        K.set_value(alpha,1)
        K.set_value(beta,0)
    print('alpha: ',K.get_value(alpha),'\nbeta: ',K.get_value(beta))
    varlam = K.variable(0.)
    
    critic_D1 = define_critic_with_gp(dD1,[*stage_dims[STAGE],out_dim[-1]],p,lrcd)
    # critic_D2 = define_critic_with_gp(dD2,[*stage_dims[STAGE],out_dim[-1]],p,lrcd)
    function_CtoD=define_composite_model(gCtoD,dD1,dD1,[*stage_dims[STAGE],inp_dim[-1]], lrgd,coord,
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
    
    print(dD1.summary())
    print(gCtoD.summary())
    
    
    # dC.set_weights(np.load('dCstage0.npy', allow_pickle=True))
    # dD.set_weights(np.load('dDstage0.npy', allow_pickle=True))
    # gCtoD.set_weights(np.load('gCtoDstage0.npy', allow_pickle=True))
    # gDtoC.set_weights(np.load('gDtoCstage0.npy', allow_pickle=True))

#%%

    while(i < steps):
        i+=1
        
        nc = options.nc if i>bat_per_epo else 2*options.nc
        print(nc)
        for iters in range(nc):
            x_realC, y_real = generate_real_samples(load_data(task,batch_size,count,
                                                            inp,tar,domain='input'), patches)
            x_realC = scale_data(x_realC, stage_dims[STAGE])            
            x_realDori, y_real = generate_real_samples(load_data(task,batch_size,count,
                                                            inp,tar,domain='target'), patches)
            x_realDori = scale_data(x_realDori, stage_dims[STAGE])
            x_realDlpf = MyFilter(coord)(x_realDori)
            x_fakeDori, y_fake = generate_fake_samples(gCtoD, x_realC, patches, out_dim[-1])
            x_fakeDori = update_image_pool(poolD, x_fakeDori)
            x_mixDori = np.zeros(x_realDori.shape);
            for b in range(x_mixDori.shape[0]):
                eps = np.random.uniform()
                x_mixDori[b] = eps*x_realDori[b] + (1.-eps)*x_fakeDori[b];
            count += batch_size
            ###################################################################
            x_fakeDlpf, y_fake = generate_fake_samples(gCtoD, MyFilter(coord)(x_realC), patches, out_dim[-1])
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
            pwidth=(np.random.uniform(.7,1.)*stage_dims[options.stage][0]).astype(np.int32)
            coord_real.append(pwidth)
            coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
            coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
        K.set_value(coord,coord_real)
        K.set_value(varlam, np.min([lam, lam*i/bat_per_epo])); print(K.get_value(varlam))
        
        if i%10==0:
            w1 = -oriLoss[1]; w2 = oriLoss[2]
            print(str('criticD r %08.3f f %08.3f' %(w1,w2)),flush=True)
            w1 = -lpfLoss[1]; w2 = lpfLoss[2]
            print(str('criticD r %08.3f f %08.3f' %(w1,w2)),flush=True)
            
        if i%10==0:
            vali = validate(task,dD1,dD1,gCtoD,evaluation,stage_dims[STAGE],MyFilter,coord)
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
            plot_curves_gp(lossD,genlossD,lam,dDval,criticW,name+'procycle')
            dif1,dif2 = evall_2(i, gCtoD, evaluation,name+'procycle',task,stage_dims[STAGE],paired=paired)
            
            if paired==False:
                dif1,dif2 = make_fid(task,evaluation,stage_dims,STAGE,count,patches,out_dim,gCtoD,inc_model)
                DIF1.append(dif1); DIF2.append(dif2)
                plt.figure()
                plt.plot(DIF1); plt.plot(DIF2)
                plt.title(str('%.3f,  %.3f' %(np.min(DIF1),np.min(DIF2))))
                plt.savefig( 'metric/%s.pdf' %(name+'procycle'),dpi=300)
            
            else:
                DIF1.append(dif1); DIF2.append(dif2)
                plt.figure()
                plt.plot(DIF1); plt.plot(DIF2)
                plt.title(str('%.3f,  %.3f' %(np.max(DIF1),np.max(DIF2))))
                plt.savefig( 'metric/%s.pdf' %(name+'procycle'),dpi=300)
        
            wg1 = gCtoD.get_weights()
            np.save(str('CtoDweights/%s_%04d' % (name,i+1)),wg1, allow_pickle=True)
            
            inp= np.random.permutation(df['input'].dropna())
            tar= np.random.permutation(df['target'].dropna())
            
        #     # wg2 = generator_DtoC.get_weights()
        #     # np.save('/scratch/shared-christoph-adela/christoph/' + 
        #     #         task + str('DtoCweights/%s_%04d' % (name,i+1)),wg2, allow_pickle=True)
            
        #     # wc = discriminator_D.get_weights()
        #     # np.save('/scratch/shared-christoph-adela/christoph/' + 
        #     #         task + str('criticDweights/%s_%04d' % (name,i+1)),wc, allow_pickle=True)
        # scale an array of images to a new size
#%%
TABLE=pd.read_csv('TABLEfinal.csv',encoding = 'unicode_escape')
if paired:
    TABLE.FID1[int(name)]=np.max(DIF1)
    TABLE.FID2[int(name)]=np.max(DIF2)
else:
    TABLE.FID1[int(name)]=np.min(DIF1)
    TABLE.FID2[int(name)]=np.min(DIF2)
    
TABLE.to_csv('TABLEfinal.csv',index=False)

for foo in range(50):
    print('###########################################################################################')
    

#%%        

# coord_real=[]
# for dummy in range(128):
#     pwidth=(np.random.uniform(.7,1.)*stage_dims[options.stage][0]).astype(np.int32)
#     coord_real.append(pwidth)
#     coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
#     coord_real.append(np.random.choice(range(stage_dims[options.stage][0]-pwidth)))
# coord = K.variable(coord_real,tf.int32)
# trans = K.variable(0.)

# x_realDori, y_real = generate_real_samples(load_data(task,128,count,
#                                                 inp,tar,domain='target'), patches)
# x_realDori = scale_data(x_realDori, stage_dims[STAGE])
# x_realDlpf = MyFilter(coord)(x_realDori)
# print(np.mean(np.abs(x_realDori-x_realDlpf)))

# ind = np.random.permutation(range(128))
# ind_i = 0; ori_list=[]; lpf_list=[]
# while ind_i < 128:
#     ori_list.append(np.mean(np.abs(x_realDori[ind[ind_i]]-x_realDori[ind[ind_i+1]])))
#     lpf_list.append(np.mean(np.abs(x_realDlpf[ind[ind_i]]-x_realDlpf[ind[ind_i+1]])))
#     ind_i +=2
    
# print(np.mean(ori_list))
# print(np.mean(lpf_list))


