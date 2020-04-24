from   . import densenet
from   . import efficientnet
from   . import senet
from   . import inception
from   . import nasnet
from   . import resnet
from   . import vgg

def get_model_fn_map():
    model_name_map = {
            'densenet169'        : densenet.densenet169,
            'densenet161'        : densenet.densenet161,
            'densenet201'        : densenet.densenet201,
            'densenet121'        : densenet.densenet121,
            'efficientnetb0'     : efficientnet.efficientnetb0,
            'efficientnetb1'     : efficientnet.efficientnetb1,
            'efficientnetb2'     : efficientnet.efficientnetb2,
            'efficientnetb3'     : efficientnet.efficientnetb3,
            'efficientnetb4'     : efficientnet.efficientnetb4,
            'efficientnetb5'     : efficientnet.efficientnetb5,
            'efficientnetb6'     : efficientnet.efficientnetb6,
            'efficientnetb7'     : efficientnet.efficientnetb7,
            'resnet50'           : resnet.resnet50,
            'resnet101'          : resnet.resnet101,
            'resnet152'          : resnet.resnet152,
            'resnet50v2'         : resnet.resnet50v2,
            'resnet101v2'        : resnet.resnet101v2,
            'resnet152v2'        : resnet.resnet152v2,
            'resnext50'          : resnet.resnext50,
            'resnext101'         : resnet.resnext101,
            'senet154'           : senet.senet154,
            'seresnext101'       : senet.seresnext101,
            'seresnext50'        : senet.seresnext50,
            'seresnet152'        : senet.seresnet152,
            'seresnet101'        : senet.seresnet101,
            'xception'           : inception.xception,
            'inceptionresnetv2'  : inception.inceptionresnetv2,
            'nasnetlarge'        : nasnet.nasnetlarge,
            'vgg16'              : vgg.vgg16,
            'vgg19'              : vgg.vgg19,
        }
    return model_name_map
# enddef
