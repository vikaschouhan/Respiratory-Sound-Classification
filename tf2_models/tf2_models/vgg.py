import tensorflow as tf
from   tensorflow.keras.applications import VGG16, VGG19 
from   tensorflow.keras import Model
from   .utils import static
from   .shared import s2s, create_model
from   .models import DenseNet161

model_map = {
            'vgg16' : VGG16,
            'vgg19' : VGG19,
        }

##########################################
# Model definition functions should define default_image_size
# vgg16
@static('default_image_size', 224)
def vgg16(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(vgg16.default_image_size, input_shape)
    return create_model('vgg16', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# Densenet169
@static('default_image_size', 224)
def vgg19(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(vgg19.default_image_size, input_shape)
    return create_model('vgg19', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
