import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.compat.v1.layers import Flatten

# from tf.nn import moments

## TODO: implement the use of spectral norm

# instNormInit = tf.initializers.random_normal(mean=1.0, stddev=0.02)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.compat.v1.image.resize_nearest_neighbor(x, size=new_size)

def l2_normalizer(v, eps=12-12):
    return v/(tf.reduce_sum(v**2)**0.5+eps)

SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS):
  # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.compat.v1.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    
    def power_iteration(i, u_i, v_i):
        v_ip1 = l2_normalizer(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = l2_normalizer(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    
    _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
                u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )

    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    
    tf.compat.v1.add_to_collection(update_collection, u.assign(u_final))
    return W_bar

def adain(content, s_mean, s_std, eps=1e-5, data_format='channels_last'):
    axes = [2,3] if data_format=='channels_first' else [1,2]

    c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
    c_std = tf.sqrt(c_var)

    s_mean = tf.expand_dims(tf.expand_dims(s_mean, 1), 1)
    s_std = tf.expand_dims(tf.expand_dims(s_std, 1), 1)

    return ((content-c_mean)/c_std)*s_std + s_mean

def convLayer(x, out_ch, kernel_size, stride, padding=None, sn=True, scope="conv0"):
    with tf.compat.v1.variable_scope(scope, reuse=False):
        k = kernel_size//2
        if padding is None:
            padding = tf.constant([[0,0], [k, k], [k, k], [0,0]])
        else:
            padding = tf.constant([[0,0], [padding, padding], [padding, padding], [0,0]])
        out = tf.pad(x, padding, "REFLECT")
        w = tf.compat.v1.get_variable("kernel", shape=[kernel_size, kernel_size, out.get_shape()[3], out_ch],
            initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
        b = tf.compat.v1.get_variable("b", [out_ch], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d(out, strides=[1,stride, stride, 1], filter=spectral_normed_weight(w), padding='VALID', data_format="NHWC")
        out = out+b
        return out

def flattenHW(x):
    return tf.reshape(x, shape=[-1, x.shape[2]*x.shape[1], x.shape[3]])

def attention(x,ch, scope='attention', reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        f = convLayer(x, ch//8, kernel_size=1, stride=1, scope= "___conv_f")
        g = convLayer(x, ch//8, kernel_size=1, stride=1, scope= "___conv_g")
        h = convLayer(x, ch, kernel_size=1, stride=1, scope= "___conv_h")

        # f = tf.keras.layers.Conv2D(ch//8, kernel_size=1, strides=1, padding="valid", data_format="channels_last", 
        #     kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02), bias_initializer=tf.constant_initializer(0.0))(x)
        # g = tf.keras.layers.Conv2D(ch//8, kernel_size=1, strides=1, padding="valid", data_format="channels_last",
        #     kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02), bias_initializer=tf.constant_initializer(0.0))(x)
        # h = tf.keras.layers.Conv2D(ch, kernel_size=1, strides=1, padding="valid", data_format="channels_last",
        #     kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02), bias_initializer=tf.constant_initializer(0.0))(x)

        s = tf.matmul(flattenHW(f), flattenHW(g), transpose_b=True)

        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, flattenHW(h))

        gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2], x.shape[3]])
        x = gamma*o + x
        
        return x

def residualBlockUp(x, out_ch, kernel_size=3, stride=1, upsample=None, scope="resUp0"):
    with tf.compat.v1.variable_scope(scope, reuse=False):
        residual = x
        if upsample:
            # residual = tf.keras.layers.UpSampling2D(size=upsample, interpolation="nearest", data_format="channels_last")(residual)
            residual = up_sample(residual, upsample)

        residual = convLayer(residual, out_ch, kernel_size=1, stride=1, scope= "___conv1")
        
        out = tf.nn.relu(tf.contrib.layers.instance_norm(x, scope= "___instanceNorm1", data_format="NHWC"))
        if upsample:
            # out =   tf.keras.layers.UpSampling2D(size=upsample, interpolation="nearest", data_format="channels_last")(out)
            out = up_sample(out, upsample)
        out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, scope= "___conv2")
        out = tf.nn.relu(tf.contrib.layers.instance_norm(out, scope= "___instanceNorm2", data_format="NHWC"))
        out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, scope= "___conv3")
        return residual + out

def residualBlockDown(x, out_ch, kernel_size=3, stride=1, padding=None, scope="resDown0"):
    with tf.compat.v1.variable_scope(scope, reuse=False):
        residual = x
        residual = convLayer(residual, out_ch, kernel_size=1, stride=1, scope= "___conv1")
        residual = tf.keras.layers.AveragePooling2D(pool_size=2)(residual)

        out = tf.nn.relu(x)
        out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, scope= "___conv2")
        out = tf.nn.relu(out)
        out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, scope= "___conv3")
        out = tf.keras.layers.AveragePooling2D(pool_size=2)(out)

        return residual + out

def residualBlock(x, channels, scope="resBlock0"):
    with tf.compat.v1.variable_scope(scope, reuse=False):
        residual = x

        out = convLayer(x, channels, kernel_size=3, stride=1, scope="___conv1")
        out= tf.nn.relu(tf.contrib.layers.instance_norm(out, scope="___instanceNorm1", data_format="NHWC"))
        out = convLayer(out, channels, kernel_size=3, stride=1, scope="___conv2")
        out= tf.contrib.layers.instance_norm(out, scope="___instanceNorm2", data_format="NHWC")

        out = tf.nn.relu(out + residual)
        return out





