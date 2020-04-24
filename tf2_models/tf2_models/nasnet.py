import tensorflow as tf
from   .utils import static
from   .shared import s2s, create_model, classification_model_init_fn

model_map = {
            'nasnetlarge'       : classification_model_init_fn('nasnetlarge'),
        }

##########################################
# Model definition functions should define default_image_size
# xception
@static('default_image_size', 331)
def nasnetlarge(num_classes, input_shape=None, weights='imagenet'):
    inp_shape = s2s(nasnetlarge.default_image_size, input_shape)
    return create_model('nasnetlarge', model_map, num_classes, input_shape=inp_shape, weights=weights)
# enddef
