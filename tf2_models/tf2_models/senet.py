import tensorflow as tf
from   .utils import static
from   .shared import s2s, create_model, classification_model_init_fn

model_map = {
            'senet154'          : classification_model_init_fn('senet154'),
            'seresnet101'       : classification_model_init_fn('seresnet101'),
            'seresnet152'       : classification_model_init_fn('seresnet152'),
            'seresnext50'       : classification_model_init_fn('seresnext50'),
            'seresnext101'      : classification_model_init_fn('seresnext101'),
        }

##########################################
# Model definition functions should define default_image_size
# Senet154
@static('default_image_size', 224)
def senet154(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(senet154.default_image_size, input_shape)
    return create_model('senet154', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def seresnext101(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(seresnext101.default_image_size, input_shape)
    return create_model('seresnext101', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def seresnext50(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(seresnext50.default_image_size, input_shape)
    return create_model('seresnext50', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def seresnet152(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(seresnet152.default_image_size, input_shape)
    return create_model('seresnet152', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 224)
def seresnet101(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(seresnet101.default_image_size, input_shape)
    return create_model('seresnet101', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
