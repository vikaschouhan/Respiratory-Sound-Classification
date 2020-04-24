import tensorflow as tf
import efficientnet.tfkeras as etf
from   tensorflow.keras import Model
from   .utils import static
from   .shared import s2s, create_model

model_map = {
            'efficientnetb0' : etf.EfficientNetB0,
            'efficientnetb1' : etf.EfficientNetB1,
            'efficientnetb2' : etf.EfficientNetB2,
            'efficientnetb3' : etf.EfficientNetB3,
            'efficientnetb4' : etf.EfficientNetB4,
            'efficientnetb5' : etf.EfficientNetB5,
            'efficientnetb6' : etf.EfficientNetB6,
            'efficientnetb7' : etf.EfficientNetB7,
        }

##########################################
# Model definition functions should define default_image_size
# EfficientNetB0
@static('default_image_size', 224)
def efficientnetb0(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb0.default_image_size, input_shape)
    return create_model('efficientnetb0', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB1
@static('default_image_size', 240)
def efficientnetb1(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb1.default_image_size, input_shape)
    return create_model('efficientnetb1', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB2
@static('default_image_size', 260)
def efficientnetb2(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb2.default_image_size, input_shape)
    return create_model('efficientnetb2', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB3
@static('default_image_size', 300)
def efficientnetb3(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb3.default_image_size, input_shape)
    return create_model('efficientnetb3', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB4
@static('default_image_size', 380)
def efficientnetb4(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb4.default_image_size, input_shape)
    return create_model('efficientnetb4', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB5
@static('default_image_size', 456)
def efficientnetb5(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb5.default_image_size, input_shape)
    return create_model('efficientnetb5', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB6
@static('default_image_size', 528)
def efficientnetb6(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb6.default_image_size, input_shape)
    return create_model('efficientnetb6', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

# EfficientNetB7
@static('default_image_size', 600)
def efficientnetb7(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(efficientnetb7.default_image_size, input_shape)
    return create_model('efficientnetb7', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
