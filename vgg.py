from keras_vggface import VGGFace
from keras.engine import Model
from keras.layers import Input,  concatenate, Reshape, Lambda, Flatten
# from tensorflow.layers import Flatten
from keras import applications
import tensorflow as tf

vgg19 = applications.vgg19.VGG19(weights="imagenet", include_top=True)
vggface = VGGFace(include_top=True)

vgg19.trainable = False
for layer in vgg19.layers:
    layer.trainable = False

vggface.trainable = False
for layer in vggface.layers:
    layer.trainable = False


# FEATURE_IDX_19 = [1, 6, 11, 20, 29]
# FEATURE_IDX_FACE = [1, 6, 11, 18, 25]
FEATURE_IDX_19 = [1, 4, 7, 12, 17]
FEATURE_IDX_FACE = [1, 4, 7, 10, 14]

vgg19_layers = [Model(vgg19.input, Flatten()(vgg19.layers[i].output)) for i in FEATURE_IDX_19]

vggface_layers = [Model(vggface.input, Flatten()(vggface.layers[i].output)) for i in FEATURE_IDX_FACE]


def vggface_output(x):
    return [m(x) for m in vggface_layers]

def vgg19_output(x):
    return [m(x) for m in vgg19_layers]

# print(model_19.summary)
# print(model_face.summary)