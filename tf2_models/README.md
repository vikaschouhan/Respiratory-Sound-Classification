# tf2_models
This is a library which provides several tensorflow2 (keras) cnn models (already available publicly) via a single coherent api. Following models are supported right now.
- densenet169
- densenet161
- densenet201
- densenet121
- efficientnetb0
- efficientnetb1
- efficientnetb2
- efficientnetb3
- efficientnetb4
- efficientnetb5
- efficientnetb6
- efficientnetb7
- resnet50
- resnet101
- resnet152
- resnet50v2
- resnet101v2
- resnet152v2
- resnext50
- resnext101
- senet154
- seresnext101
- seresnext50
- seresnet152
- seresnet101
- xception
- inceptionresnetv2
- nasnetlarge
- vgg16
- vgg19

# Usage
```python
import tf2_models
init_fn = tf2_models.get_model_fn_map()
model   = init_fn['densenet161'](num_classes=10)
```

# Model Api
```
model = model_init_fn(num_classes, input_shape)
```
