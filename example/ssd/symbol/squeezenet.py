import mxnet as mx

def fire_module(data, squeeze_depth, expand_depth, prefix):
    fire_squeeze1x1 = mx.symbol.Convolution(name='{}_squeeze1x1'.format(prefix), data=data, num_filter=squeeze_depth, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    fire_relu_squeeze1x1 = mx.symbol.Activation(name='{}_relu_squeeze1x1'.format(prefix), data=fire_squeeze1x1, act_type='relu')
    fire_expand1x1 = mx.symbol.Convolution(name='{}_expand1x1'.format(prefix), data=fire_relu_squeeze1x1, num_filter=expand_depth, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    fire_relu_expand1x1 = mx.symbol.Activation(name='{}_relu_expand1x1'.format(prefix), data=fire_expand1x1, act_type='relu')
    fire_expand3x3 = mx.symbol.Convolution(name='{}_expand3x3'.format(prefix), data=fire_relu_squeeze1x1, num_filter=expand_depth, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    fire_relu_expand3x3 = mx.symbol.Activation(name='{}_relu_expand3x3'.format(prefix), data=fire_expand3x3, act_type='relu')
    fire_concat = mx.symbol.Concat(name='{}_concat'.format(prefix), *[fire_relu_expand1x1, fire_relu_expand3x3])

    return fire_concat

def squeezenet_v10(num_classes=1000):
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=96, pad=(0,0), kernel=(7,7), stride=(2,2), no_bias=False)
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_conv1, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')

    fire2_concat = fire_module(pool1, 16, 64, "fire2")
    fire3_concat = fire_module(fire2_concat, 16, 64, "fire3")
    fire4_concat = fire_module(fire3_concat, 32, 128, "fire4")
    pool4 = mx.symbol.Pooling(name='pool4', data=fire4_concat, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')

    fire5_concat = fire_module(pool4, 32, 128, "fire5")
    fire6_concat = fire_module(fire5_concat, 48, 192, "fire6")
    fire7_concat = fire_module(fire6_concat, 48, 192, "fire7")
    fire8_concat = fire_module(fire7_concat, 64, 256, "fire8")
    pool8 = mx.symbol.Pooling(name='pool8', data=fire8_concat, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
    fire9_concat = fire_module(pool8, 64, 256, "fire9")
    drop9 = mx.symbol.Dropout(name='drop9', data=fire9_concat, p=0.500000)

    conv10 = mx.symbol.Convolution(name='conv10', data=drop9, num_filter=num_classes, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu_conv10 = mx.symbol.Activation(name='relu_conv10', data=conv10, act_type='relu')
    pool10 = mx.symbol.Pooling(name='pool10', data=relu_conv10, pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=flatten)

    return softmax

def squeezenet_v11(num_classes=1000):
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(0,0), kernel=(3,3), stride=(2,2), no_bias=False)
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_conv1, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')

    fire2_concat = fire_module(pool1, 16, 64, "fire2")
    fire3_concat = fire_module(fire2_concat, 16, 64, "fire3")
    pool3 = mx.symbol.Pooling(name='pool3', data=fire3_concat, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')

    fire4_concat = fire_module(pool3, 32, 128, "fire4")
    fire5_concat = fire_module(fire4_concat, 32, 128, "fire5")
    pool5 = mx.symbol.Pooling(name='pool5', data=fire5_concat, pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')

    fire6_concat = fire_module(pool5, 48, 192, "fire6")
    fire7_concat = fire_module(fire6_concat, 48, 192, "fire7")
    fire8_concat = fire_module(fire7_concat, 64, 256, "fire8")
    fire9_concat = fire_module(fire8_concat, 64, 256, "fire9")
    drop9 = mx.symbol.Dropout(name='drop9', data=fire9_concat, p=0.500000)

    conv10 = mx.symbol.Convolution(name='conv10', data=drop9, num_filter=num_classes, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu_conv10 = mx.symbol.Activation(name='relu_conv10', data=conv10, act_type='relu')
    pool10 = mx.symbol.Pooling(name='pool10', data=relu_conv10, pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=flatten)

    return softmax

def get_symbol(num_classes=1000, version="v1.0", **kwargs):
    assert version == "v1.0" or version == "v1.1"

    if version == "v1.0":
        return squeezenet_v10(num_classes)
    else:
        return squeezenet_v11(num_classes)
    