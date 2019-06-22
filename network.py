
from .components import residualBlock, residualBlockUp, residualBlockDown, adain, attention, convLayer
import tensorflow as tf
import keras as K
from collections import OrderedDict

# TODO: wieight inits

# class Emebedder(object):

#     def __init__():
#         res1 = 

def embedder(x, y):
    assert x.shape == y.shape
    out = tf.concat(x, y, axis=0)
    out = out.expand_dims(out, 0)

    out = tf.nn.relu(residualBlockDown(out, 64))
    out = tf.nn.relu(residualBlockDown(out, 128))
    out = tf.nn.relu(residualBlockDown(out, 256))
    out = attention(out, 256)
    out = tf.nn.relu(residualBlockDown(out, 512))
    
    out = tf.nn.relu(K.layers.GlobalMaxPooling2D(data_format="channels_first")(out).reshape(512, 1))

    return out

def generator(y, e):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        assert(y.shape) == [3,224, 224]
        psi_width = 32
        adain_layers = OrderedDict([
            ('deconv4', 256),
            ('deconv3', 128),
            ('deconv2', 64),
            ('deconv1', 3)
        ])
        
        out = {}
        start, end = 0, 0
        for layer in adain_layers:
            end = start + d[layer]*2
            out[layer] = (start, end)
            start = end
        projection = tf.get_variable("projection", [psi_length, 512])

        def slice_psi(psi, portion):
            start, end = out[portion]
            a = tf.reshape(psi[start:end], (2, -1))
            return tf.reshape(a[0], [1,-1,1,1]),  tf.reshape(a[1], [1,-1,1,1])

        out = tf.expand_dims(y, 0)

        out = tf.nn.relu(tf.contrib.instance_norm(residualBlockDown(out, 64)))
        out = tf.nn.relu(tf.contrib.instance_norm(residualBlockDown(out, 128)))
        out = tf.nn.relu(tf.contrib.instance_norm(residualBlockDown(out, 256)))
        out = attention(out, 256)
        out = tf.nn.relu(tf.contrib.instance_norm(residualBlockDown(out, 512)))
        
        out = residualBlock(out, 512)
        out = residualBlock(out, 512)
        out = residualBlock(out, 512)
        out = residualBlock(out, 512)
        out = residualBlock(out, 512)

        psi_hat = tf.reshape(tf.matmul(projection, e), -1)

        out = residualBlockUp(out, 256, upsample=2)
        out = tf.nn.relu(adain(out, *slice_psi(psi_hat, 'deconv4')))
        out = residualBlockUp(out, 128, upsample=2)
        out = tf.nn.relu(adain(out, *slice_psi(psi_hat, 'deconv3')))
        
        out = attention(out, 128)

        out = residualBlockUp(out, 64, upsample=2)
        out = tf.nn.relu(adain(out, *slice_psi(psi_hat, 'deconv2')))
        out = residualBlockUp(out, 3, upsample=2)
        out = tf.nn.relu(adain(out, *slice_psi(psi_hat, 'deconv1')))

    return out[0]


def discriminator(ntv, x, y, i):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        out = tf.concat(x, y, axis=0)
        out = tf.expand_dims(out, 0)
        
        out0 = tf.nn.relu(residualBlockDown(out, 64))
        out1 = tf.nn.relu(residualBlockDown(out0, 128))
        out2 = tf.nn.relu(residualBlockDown(out1, 256))
        out3 = attention(out2, 256)
        out4 = tf.nn.relu(residualBlockDown(out3, 512))

        out = residualBlock(out4, 512)

        out = tf.nn.relu(K.layers.GlobalMaxPooling2D(data_format="channels_first")(out).reshape(512, 1))

        W = tf.get_variable("W", shape=(512, ntv), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        w0 = tf.get_variable("w0", shape=(512, 1), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        b = tf.get_variable("b", shape=(1,), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))

        tf.matmul(out, tf.slice())


