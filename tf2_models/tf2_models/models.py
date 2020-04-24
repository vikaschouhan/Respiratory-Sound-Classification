from   tensorflow.keras.models import Model
from   tensorflow.keras.layers import Input, concatenate, ZeroPadding2D
from   tensorflow.keras.layers import Dense, Dropout, Activation
from   tensorflow.keras.layers import Convolution2D
from   tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from   tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
from   tensorflow.keras.utils import get_file
from   .custom_layers import Scale

DENSENET_161_WEIGHTS_PATH = (r'https://github.com/vikaschouhan/DenseNet-Keras/releases/'
                             r'download/v1.0/densenet161_weights_tf.h5')

# Define densenet161
def DenseNet161(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling='avg',
             classes=1000):
    if not (weights in {'imagenet', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # endif

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # endif
    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"`, `classes` should be 1000')
    # endif
    if pooling not in ['avg', 'max']:
        raise ValueError('pooling should be `avg` or `max`')
    # endif

    load_weights = True if weights == 'imagenet' else False
    input_shape  = (224, 224, 3) if input_shape is None else input_shape

    return __DenseNet161(classes=classes, include_top=include_top,
            load_weights=load_weights, input_tensor=input_tensor,
            input_shape=input_shape, pooling=pooling)
# enddef

def __DenseNet161(nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.0,
        weight_decay=1e-4, load_weights=True, include_top=False, input_tensor=None,
        pooling='avg', input_shape=(224, 224, 3), classes=1000):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        # endif
    # endif

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_data_format() == 'channels_last':
        concat_axis = 3
    else:
        concat_axis = 1
    # endif

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    # endfor

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    if include_top:
        x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='pool'+str(final_stage))(x)
        # endif
    # endif
    fc_ll = x

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet161')
    if load_weights:
        weights_path = get_file('densenet161_weights_tf.h5', DENSENET_161_WEIGHTS_PATH, cache_subdir='models')
        model.load_weights(weights_path)
    # endif

    if include_top == False:
        model = Model(img_input, fc_ll)
    # endif

    return model
# enddef


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, kernel_size=(1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    # endif

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, kernel_size=(3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    # endif

    return x
# enddef


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), kernel_size=(1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    # endif

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x
# enddef


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate
        # endif
    # endfor

    return concat_feat, nb_filter
# enddef
