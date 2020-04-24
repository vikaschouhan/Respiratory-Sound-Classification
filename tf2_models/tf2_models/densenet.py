import tensorflow as tf
from   tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from   tensorflow.keras import Model
from   .utils import static
from   .shared import s2s, create_model
from   .models import DenseNet161

model_map = {
            'densenet169' : DenseNet169,
            'densenet121' : DenseNet121,
            'densenet201' : DenseNet201,
            'densenet161' : DenseNet161,
        }

##########################################
# Model definition functions should define default_image_size
# Densenet121
@static('default_image_size', 224)
def densenet121(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(densenet121.default_image_size, input_shape)
    return create_model('densenet121', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# Densenet169
@static('default_image_size', 224)
def densenet169(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(densenet169.default_image_size, input_shape)
    return create_model('densenet169', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# Densenet201
@static('default_image_size', 224)
def densenet201(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(densenet201.default_image_size, input_shape)
    return create_model('densenet201', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# Densenet161
@static('default_image_size', 224)
def densenet161(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(densenet161.default_image_size, input_shape)
    return create_model('densenet161', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
