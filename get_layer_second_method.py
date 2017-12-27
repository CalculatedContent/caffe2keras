from keras.activations import relu, softmax
from keras.models import Input, Model
from keras.layers import Dense, BatchNormalization, Conv2D, Activation, \
Dropout, Add, MaxPooling2D, AveragePooling2D, Flatten, Concatenate
from keras import backend as K
import numpy as np

def get_bottom(bottom_name, layers, debug_layers=None):
    
    for key in layers.keys():
        if bottom_name == key:
            bottom = layers[key]
            
            debug_layers['layer_connect'][key] = debug_layers['layer_connect'].get(key, 0) + 1

            return bottom

    if bottom_name=='data':
        if K.backend()=='tensorflow':
            input_dim = (None, None, 3)
        else:
            input_dim = (3, None, None)
        layer = Input(input_dim, name='data')
        layers['data'] = layer
        return layer
        
    print('KEY ERROR:please load bottom layer: {} before this layer'.format(bottom_name))  

def get_layer(layer_params, layers, verbose=False, debug_layers=None):   
    try:      
        layer_name = layer_params['name'] 
        layer_type = layer_params['type'] 

        if layer_name not in debug_layers.keys():
            debug_layers['layer_connect'][layer_name] = 0
        if verbose:
            print('loading: ', layer_name)
            # print('layer params:\n', layer_params)
    except:
        if verbose:
            print('loading failed')
            print('\nlayer params:\n', layer_params)
        return None

#Input layer
    if layer_type == 'input':
        try:
            input_dim = np.array(layer_params['input_dim'][1:])
            if K.backend()=='tensorflow':
                input_dim = tuple(input_dim[[1, 2, 0]])
        except:
            if K.backend()=='tensorflow':
                input_dim = (None, None, 3)
            else:
                input_dim = (3, None, None)
        layer = Input(input_dim, name='data')
        layers[layer_name] = layer

        return layer 
    
#Convolution layer
    elif layer_type == 'convolution':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        # print(bottom_name,' \n', bottom)
        kernel_size = layer_params['kernel_size']
        num_output = layer_params['num_output']
        pad = layer_params.get('pad', 0)
        stride = layer_params.get('stride', 1)
        use_bias = layer_params.get('bias_term', True)
        padding = 'same' if pad==(kernel_size-1)/2 else 'valid'
        layer = Conv2D(num_output, kernel_size, strides=(stride, stride), padding=padding, name=layer_name,
                       use_bias=use_bias)(bottom)
        layers[layer_name] = layer
        
        return layer
    
#Dense layer
    elif layer_type == 'inner_product' or layer_type=='innerproduct':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        num_output = layer_params['num_output']
        bottom_dim = bottom.shape.as_list()
        if len(bottom_dim)!=2:
            bottom = Flatten()(bottom)
        layer = Dense(num_output, name=layer_name)(bottom)
        layers[layer_name] = layer
        return layer
    
#Batch_norm layer
    elif layer_type == 'batchnorm':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
#         num_output = layer_params['num_output']
        layer = BatchNormalization(name=layer_name)(bottom)
        layers[bottom_name] = layer
        return layer
    
#Pool layer
    elif layer_type == 'pooling':
        bottom_name = layer_params['bottom'][0]
        pool = layer_params['pool']
        stride = layer_params.get('stride', 2)
        kernel_size = layer_params.get('kernel_size', 2)
        bottom = get_bottom(bottom_name, layers, debug_layers)
        if pool=='max':
            layer = MaxPooling2D(pool_size=(kernel_size,kernel_size), strides=(stride,stride),
                                 name=layer_name)(bottom)
        elif pool=='ave':
            layer = AveragePooling2D(pool_size=(kernel_size,kernel_size), strides=(stride,stride),
                                     name=layer_name)(bottom)
        layers[layer_name] = layer
        return layer
    
#Softmax layer
    elif 'softmax' in layer_type:
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        layer = Activation(softmax, name=layer_name)(bottom)
#         layers[layer_name] = layer
        layers[bottom_name] = layer
        return layer
    
    elif layer_type == 'relu':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        layer = Activation(relu, name=layer_name)(bottom)
#         layers[layer_name] = layer
        layers[bottom_name] = layer
        return layer

#Dropout layer
    elif layer_type == 'dropout':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        dropout = 1 - float(layer_params['dropout_ratio'])
        layer = Dropout(dropout, name=layer_name)(bottom)
#         layers[layer_name] = layer
        layers[bottom_name] = layer
        return layer
    
#Add layer
    elif layer_type == 'eltwise':
        bottom_name1 = layer_params['bottom'][0]
        bottom_name2 = layer_params['bottom'][1]
        bottom1 = get_bottom(bottom_name1, layers, debug_layers)
        bottom2 = get_bottom(bottom_name2, layers, debug_layers)
        
        layer = Add(name=layer_name)([bottom1, bottom2])
        layers[layer_name] = layer
        return layer
 
#Concat layer
    elif layer_type == 'concat':
        bottom_name1 = layer_params['bottom'][0]
        bottom_name2 = layer_params['bottom'][1]
        bottom1 = get_bottom(bottom_name1, layers, debug_layers)
        bottom2 = get_bottom(bottom_name2, layers, debug_layers)
        
        layer = Concatenate(name=layer_name)([bottom1, bottom2])
        layers[layer_name] = layer
        return layer

    elif layer_type == 'lrn':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers, debug_layers)
        layer= BatchNormalization()(bottom)
        layers[layer_name] = layer
        return layer

    #Ignore data layer
    elif layer_type == 'data':
        return None

    else:
        debug_layers['skip_layer'].append(layer_name)
        if verbose:
            print("skipped: \"{}\" layer, please check the model later, may be it's not right builded".format(layer_type))