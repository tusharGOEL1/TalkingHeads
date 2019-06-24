import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from components import residualBlock, residualBlockUp, residualBlockDown, adain, attention, convLayer
import tensorflow as tf
import keras as K
from collections import OrderedDict
# TODO: wieight inits


def embedder(x, y, scope="embedder"):
    with tf.compat.v1.variable_scope(scope):

        out = tf.concat([x, y], axis=3)
        out = tf.nn.relu(residualBlockDown(out, 64, scope= "___resDown1"))
        out = tf.nn.relu(residualBlockDown(out, 128, scope= "___resDown2"))
        out = tf.nn.relu(residualBlockDown(out, 256, scope= "___resDown3"))
        out = attention(out, 256, scope= "___attn")
        out = tf.nn.relu(residualBlockDown(out, 512, scope= "___resDown4"))
        
        out = tf.nn.relu(K.layers.GlobalMaxPooling2D(data_format="channels_last")(out))
        return out

def generator(y, e, scope="generator"):
    with tf.compat.v1.variable_scope(scope, reuse=False):

        adain_layers = OrderedDict([
            ('deconv4', 256),
            ('deconv3', 128),
            ('deconv2', 64),
            ('deconv1', 3)
        ])
        
        out = {}
        start, end = 0, 0
        for layer in adain_layers:
            end = start + adain_layers[layer]*2
            out[layer] = (start, end)
            start = end
        projection = tf.compat.v1.get_variable("projection", [end, 512], dtype=tf.float32)

        out = y

        out = tf.nn.relu(tf.contrib.layers.instance_norm(residualBlockDown(out, 64, scope= "___resDown1"), data_format="NHWC"))
        out = tf.nn.relu(tf.contrib.layers.instance_norm(residualBlockDown(out, 128, scope= "___resDown2"), data_format="NHWC"))
        out = tf.nn.relu(tf.contrib.layers.instance_norm(residualBlockDown(out, 256, scope= "___resDown3"), data_format="NHWC"))
        out = attention(out, 256, scope= "___attn1")
        out = tf.nn.relu(tf.contrib.layers.instance_norm(residualBlockDown(out, 512, scope= "___resDown4"), data_format="NHWC"))

        out = residualBlock(out, 512, scope= "___resBlock1")
        out = residualBlock(out, 512, scope= "___resBlock2")
        out = residualBlock(out, 512, scope= "___resBlock3")
        out = residualBlock(out, 512, scope= "___resBlock4")
        out = residualBlock(out, 512, scope= "___resBlock5")

        psi_hat = tf.matmul(e, projection, transpose_b=True)

        out = residualBlockUp(out, 256, upsample=2, scope= "___resUp1")
        out = tf.nn.relu(adain(out, tf.slice(psi_hat, [0, 0], [-1, 256]), tf.slice(psi_hat, [0, 256], [-1, 256])))
        out = residualBlockUp(out, 128, upsample=2, scope= "___resUp2")
        out = tf.nn.relu(adain(out, tf.slice(psi_hat, [0, 512], [-1, 128]), tf.slice(psi_hat, [0, 640], [-1, 128])))
        
        out = attention(out, 128, scope= "___attn2")

        out = residualBlockUp(out, 64, upsample=2, scope= "___resUp3")
        out = tf.nn.relu(adain(out, tf.slice(psi_hat, [0, 768], [-1, 64]), tf.slice(psi_hat, [0, 832], [-1, 64])))
        out = residualBlockUp(out, 3, upsample=2, scope= "___resUp4")
        out = tf.nn.relu(adain(out, tf.slice(psi_hat, [0, 896], [-1, 3]), tf.slice(psi_hat, [0, 899], [-1, 3])))

    return out


def discriminator(ntv, x, y, e=None, i=None,scope="discriminator"):
    with tf.compat.v1.variable_scope(scope, reuse=False):
        out = tf.concat([x, y], axis=3)
        # out = tf.expand_dims(out, 0)
        
        out0 = tf.nn.relu(residualBlockDown(out, 64, scope= "___resDown1"))
        out1 = tf.nn.relu(residualBlockDown(out0, 128, scope= "___resDown2"))
        out2 = tf.nn.relu(residualBlockDown(out1, 256, scope= "___resDown3"))
        out3 = attention(out2, 256, scope= "___attn")
        out4 = tf.nn.relu(residualBlockDown(out3, 512, scope= "___resDown4"))

        out = residualBlock(out4, 512, scope= "___resBlock1")

        out = tf.nn.relu(K.layers.GlobalMaxPooling2D(data_format="channels_last")(out))
        print(out.shape)
        W = tf.compat.v1.get_variable( "___W", shape=(ntv, 512,), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        w0 = tf.compat.v1.get_variable( "___w0", shape=(1, 512), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        b = tf.compat.v1.get_variable( "___b", shape=(1,), initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        if i is None:
            out = tf.reduce_sum(e*out, axis=1) + b
        else:
            out = tf.reduce_sum((tf.nn.embedding_lookup(W,i)+ w0)*out, axis=1) + b

        out = tf.nn.tanh(out)
        return out, [out0, out1, out2, out3, out4]




if __name__ == "__main__":
    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
    e = tf.compat.v1.placeholder(tf.float32, [None, 512])
    i = tf.compat.v1.placeholder(tf.int32, [None])
    # e = embedder(x,y)

    # g = generator(y, e)
    d = discriminator(1000, x, y, e, i)
