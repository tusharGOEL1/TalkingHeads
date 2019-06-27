import tensorflow as tf
import keras
from vgg import vggface_output, vgg19_output

def L_cnt(x_real, x_fake):
    loss = tf.constant(0.0)
    vr = vggface_output(x_real)
    vf = vggface_output(x_fake)
    for r,f in zip(vr, vf):
        loss += tf.reduce_mean(tf.math.abs(r-f), axis=1)
    
    
    vr = vgg19_output(x_real)
    vf = vgg19_output(x_fake)
    for r,f in zip(vr, vf):
        loss += tf.reduce_mean(tf.math.abs(r-f), axis=1)
    
    return loss
    
def L_adv(r_fake, D_real, D_fake):
    loss = -r_fake
    for r,f in zip(D_fake, D_real):
        loss += tf.reduce_mean(tf.math.abs(r-f), axis=1)
    return loss

def L_mch(e, W, i):
    Wi = tf.nn.embedding_lookup(W,i)
    return tf.reduce_mean(tf.math.abs(e-Wi), axis=1)

def lossGenerator(x_real, x_fake, r_fake, e, W, D_real, D_fake, i):
    return L_cnt(x_real, x_fake) + L_adv(r_fake, D_real, D_fake) + L_mch(e, W, i)

def lossDiscriminator(r_real, r_fake):
    return tf.math.maximum(0.0, 1.0+r_fake) + tf.math.maximum(0.0, 1.0-r_real)


if __name__ == '__main__':
    x_real = tf.placeholder(tf.float32, [None, 224, 224, 3])
    x_fake = tf.placeholder(tf.float32, [None, 224, 224, 3])
    shapes = [(None, 112*112*64),
                (None, 56*56*128),
                (None, 28*28*256),
                (None, 28*28*256),
                (None, 14*14*512)]
    D_real = [tf.placeholder(tf.float32, shape) for shape in shapes]
    D_fake = [tf.placeholder(tf.float32, shape) for shape in shapes]
    r_fake = tf.placeholder(tf.float32, [None])
    r_real = tf.placeholder(tf.float32, [None])
    e = tf.placeholder(tf.float32, [None, 512])
    W = tf.placeholder(tf.float32, [None, 512])
    i = tf.placeholder(tf.float32, [None])

    L_cnt(x_real, x_fake)
    L_adv(r_fake, D_real, D_fake)
    L_mch(e, W, i)
    lossGenerator(x_real, x_fake, r_fake, e, W, D_real, D_fake, i)
    lossDiscriminator(r_real, r_fake)
