from network import *
from loss import *
import tensorflow as tf
import os
import time
import tqdm
import scipy.io as sio
from PIL import Image
import numpy as np
import pickle
from utils import plot_landmarks

K=8

meanScale, stdScale = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def normalize(img):
    m = np.mean(img, axis=(0,1), keepdims=True)
    s = np.std(img, axis=(0,1), keepdims=True)
    return (stdScale* (img-m))/s + meanScale

def meta_train(datapath=None, epochs=10, size=0):
    if datapath is None:
        return
    files= [os.path.join(path, filename)
        for path, dirs, files in os.walk(datapath)
        for filename in files
        if filename.endswith('.vid')
    ]
    if not size==0:
        size = min(len(files), size)
    else:
        size = min(len(files))

    files = files[:size]
    frames_p = []
    x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name='x_t')
    y = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name='y_t')
    i_p = tf.placeholder(tf.float32, [1])
    for i in range(K):
        frames_p.append([tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name='x'+str(i)), tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name='y'+str(i))])
    embedder = Embedder()
    generator = Generator()
    discriminator = Discriminator(size)
    e_h = tf.reduce_mean(tf.concat([embedder(frames_p[i][0], frames_p[i][1]) for i in range(K)], axis=0), axis=0, keepdims=True)
    g = generator(y, e_h)
    r, d = discriminator(x, y, e_h, i_p)
    r_hat, d_hat = discriminator(g, y, e_h, i)
    loss_ge = lossGenerator(x, g, r_hat, e, discriminator.get_W(), d, d_hat, i)
    loss_d = lossDiscriminator(r, r_hat)
    GE_opt = tf.train.AdamOptimizer(loss_ge, var_list = generator.var_list()+embedder.var_list())
    D_opt = tf.train.AdamOptimizer(loss_d, var_list = discriminator.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for file in files:
        dct = pickle.load(open(file, 'rb'))
        for i in range(8):
            dct[i]['landmark'] = plot_landmarks(dct[i]['frame'], dct[i]['landmark'])
            dct[i]['landmark'] = normalize(dct[i]['landmark'])
            dct[i]['frame'] = normalize(dct[i]['frame'])
        