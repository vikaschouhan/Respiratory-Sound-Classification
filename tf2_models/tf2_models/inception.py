import tensorflow as tf
from   .utils import static
from   .shared import s2s, create_model, classification_model_init_fn

model_map = {
            'xception'          : classification_model_init_fn('xception'),
            'inceptionresnetv2' : classification_model_init_fn('inceptionresnetv2'),
        }

##########################################
# Model definition functions should define default_image_size
# xception
@static('default_image_size', 299)
def xception(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(xception.default_image_size, input_shape)
    return create_model('xception', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef

@static('default_image_size', 299)
def inceptionresnetv2(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(inceptionresnetv2.default_image_size, input_shape)
    return create_model('inceptionresnetv2', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
