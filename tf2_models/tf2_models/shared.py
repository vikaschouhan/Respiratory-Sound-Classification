import tensorflow as tf
from   tensorflow.keras.layers import Dense, Flatten, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D
from   tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from   tensorflow.keras import Model
from   distutils.version import LooseVersion
from   .utils import static
from   classification_models.tfkeras import Classifiers

# Some standard names
LAST_FC_LAYER_NAME = 'fc1'

# size to shape
def s2s(size, inp_shape=None):
    return net_size_to_shape(size) if inp_shape is None else inp_shape
# enddef

# Common function for adding last fc layer
def add_final_layer(inp_layer, num_classes, activation='softmax'):
    out_1 = Dense(num_classes, activation=None, name=LAST_FC_LAYER_NAME)(inp_layer)
    out_2 = Activation(activation=activation, name=activation)(out_1)
    return out_2
# enddef

# Shared model creator
def create_model(model_type, model_map, num_classes, input_shape, pooling='avg', weights='imagenet'):
    # Check
    assert model_type in model_map.keys(), 'model_type {} not supported in {}'.format(model_type, model_map.keys())

    model_backbone = model_map[model_type](input_shape=input_shape, weights=weights, include_top=False, pooling=pooling)
    model_out      = add_final_layer(model_backbone.output, num_classes, activation='softmax')

    model_this     = Model(inputs=model_backbone.inputs, outputs=model_out)
    # Add user defined functions
    model_this     = keras_model_wrapper(model_this)
    return model_this
# enddef

####################################################
# Model wrapper with few additional utility functions
# NOTE: Not using a derived class of keras.Model, since
#       using custom layers and objects create a lot of issues
#       while saving/restoring checkpoints. Better to 
#       not use classes for adding functionality at this time
def keras_model_wrapper(model):
    # Get logits layer
    def logits(self):
        for layer in self.layers:
            layer_cfg = layer.get_config()
            if layer_cfg['name'] == LAST_FC_LAYER_NAME:
                return self.get_layer(layer.name)
            # endif
        # endfor
        raise Exception('No logits found.')
    # enddef

    def prelogits(self):
        logit_layer_name = self.logits().get_config()['name']
        for indx, layer in enumerate(self.layers):
            layer_cfg = layer.get_config()
            if layer_cfg['name'] == logit_layer_name:
                return self.layers[indx-1]
            # endif
        # endfor
        raise Exception('No prelogits found.')
    # enddef

    # Get softmax layer
    def softmax(self):
        for layer in self.layers:
            layer_cfg = layer.get_config()
            if 'activation' in layer_cfg and layer_cfg['activation'] == 'softmax':
                return self.get_layer(layer.name)
            # endif
        # endfor
        raise Exception('No softmax found.')
    # enddef

    # Get list of layer names
    def layer_names(self):
        layer_names = [x.name for x in self.layers]
        return layer_names
    # enddef

    # Add functions
    model.__class__.logits          = logits
    model.__class__.softmax         = softmax
    model.__class__.layer_names     = layer_names
    model.__class__.prelogits       = prelogits

    return model
# endclass

####################
def net_size_to_shape(size, channels=3):
    return (size, size, channels)
# enddef

def classification_model_init_fn(model_type):
    # Add pooling layer.
    # By default "pooling" param is not supported by classification_models
    def __model_wrap(input_shape, weights='imagenet', include_top=False, pooling='avg'):
        if pooling not in ['avg', 'max']:
            raise ValueError('pooling should be `avg` or `max`')
        # endif
        # Initialize base model
        model_b = Classifiers.get(model_type)[0](
                    input_shape = input_shape,
                    weights     = weights,
                    include_top = include_top,
                    pooling     = None
                )
        # Add pooling layer
        if pooling == 'avg':
            model_out = GlobalAveragePooling2D(name='avgpool')(model_b.output)
        elif pooling == 'max':
            model_out = GlobalMaxPooling2D(name='maxpool')(model_b.output)
        # endif
        # Create new model after pooling
        model_this = Model(inputs=model_b.inputs, outputs=model_out)
        return model_this
    # enddef

    return __model_wrap
# enddef
