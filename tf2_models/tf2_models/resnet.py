import tensorflow as tf
from   .utils import static
from   .shared import s2s, create_model, classification_model_init_fn

model_map = {
            'resnet50'          : classification_model_init_fn('resnet50'),
            'resnet101'         : classification_model_init_fn('resnet101'),
            'resnet152'         : classification_model_init_fn('resnet152'),
            'resnet50v2'        : classification_model_init_fn('resnet50v2'),
            'resnet101v2'       : classification_model_init_fn('resnet101v2'),
            'resnet152v2'       : classification_model_init_fn('resnet152v2'),
            'resnext50'         : classification_model_init_fn('resnext50'),
            'resnext101'        : classification_model_init_fn('resnext101'),
        }

##########################################
# Model definition functions should define default_image_size
# Senet154
@static('default_image_size', 224)
def resnet50(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet50.default_image_size, input_shape)
    return create_model('resnet50', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnet101(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet101.default_image_size, input_shape)
    return create_model('resnet101', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnet152(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet152.default_image_size, input_shape)
    return create_model('resnet152', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnet50v2(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet50v2.default_image_size, input_shape)
    return create_model('resnet50v2', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnet101v2(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet101v2.default_image_size, input_shape)
    return create_model('resnet101v2', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnet152v2(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnet152v2.default_image_size, input_shape)
    return create_model('resnet152v2', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnext50(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnext50.default_image_size, input_shape)
    return create_model('resnext50', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def resnext101(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(resnext101.default_image_size, input_shape)
    return create_model('resnext101', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
