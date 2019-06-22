import tensorflow as tf
# from tf.nn import moments

## TODO: implement the use of spectral norm

def l2_normalizer(v, eps=12-12):
    return v/(tf.reduce_sum(v**2)**0.5+eps)

SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS):
  # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    
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
    
    tf.add_to_collection(update_collection, u.assign(u_final))
    return W_bar

def adain(content, s_mean, s_std, eps=1e-5, data_format='channel_first'):
    # assertlen(content.size()) >= 3
    axes = [2,3] if data_format=='channel_first' else [1,2]
    c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
    #s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
    # c_std, s_std = tf.sqrt(c_var), tf.sqrt(s_var)
    c_std = tf.sqrt(c_var)
    
    return ((content-c_mean)/c_std)*s_std + s_mean

def convLayer(x, out_ch, kernel_size, stride, padding=None, sn=True, scope="conv0"):
    with tf.variable_scope(scope):
        k = kernel_size//2
        if padding is None:
            padding = tf.constant([[k, k], [k, k]])
        else:
            padding = tf.constant([[padding, padding], [padding, padding]])
        out = tf.pad(x, padding, "REFLECT")
        w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, out.get_Shape()[1], out_ch],
            initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.conv2d(out, strides=[1,1,stride, stride], filter=spectral_normed_weight(w), padding='SAME', data_format="NCHW")
        return out

def flattenHW(x):
    return tf.reshape(x, shape=[x.shape[0], x.shape[1], -1])

def attention(x,ch, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = tf.keras.layers.Conv2D(ch//8, kernel_size=1, stride=1, padding="valid") (x, ch//8, kernel_size=1, data_format="channels_first")(x)
        g = tf.keras.layers.Conv2D(ch//8, kernel_size=1, stride=1, padding="valid") (x, ch//8, kernel_size=1, data_format="channels_first")(x)
        h = tf.keras.layers.Conv2D(ch, kernel_size=1, stride=1, padding="valid") (x, ch//8, kernel_size=1, data_format="channels_first")(x)

        s = tf.matmul(flattenHW(g), flattenHW(f), transpose_b=True)

        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, flattenHW(h))

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.shape)
        x = gamma*o + x
        
        return x

def residualBlockUp(x, out_ch, kernel_size=3, stride=1, upsample=None):
    residual = x
    residual = tf.keras.layers.UpSampling2D(size=upsample, interpolation="nearest", data_format="NCHW")(residual)
    residual = convLayer(residual, out_ch, kernel_size=1, stride=1)
    
    out = tf.nn.relu(tf.contrib.instance_norm(x))
    if upsample:
        out =   tf.keras.layers.UpSampling2D(size=upsample, interpolation="nearest", data_format="NCHW")(out)
    out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride)
    out = tf.nn.relu(tf.contrib.instance_norm(out))
    out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride)
    
    return residual + out

def residualBlockDown(x, out_ch, kernel_size=3, stride=1, padding=None):
    residual = x
    residual = convLayer(residual, out_ch, kernel_size=1, stride=1)
    residual = tf.keras.layers.AveragePooling2D(pool_size=2)(residual)

    out = tf.nn.relu(x)
    out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    out = tf.nn.relu(out)
    out = convLayer(out, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    out = tf.keras.layers.AveragePooling2D(pool_size=2)(out)

    return residual + out

def residualBlock(x, channels):
    residual = x
    out= tf.nn.relu(tf.contrib.layers.instance_norm(convLayer(x, channels, kernel_size=3, stride=1)))
    out= tf.contrib.layers.instance_norm(convLayer(x, channels, kernel_size=3, stride=1))
    out = tf.nn.relu(out + residual)
    return out

















