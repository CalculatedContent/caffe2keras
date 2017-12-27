
# coding: utf-8

# block some warning about cpu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#---------------------------------------

import caffe
from tqdm import tqdm
import numpy as np
# from keras.activations import relu, softmax
from keras.models import Input, Model
# from keras.layers import Dense, BatchNormalization, Conv2D, Activation, Dropout, Add, MaxPooling2D, AveragePooling2D, Flatten
from keras import backend as K
import re

from get_layer_second_method import get_layer

def model_preprocessing(caffe_model):
    with open(caffe_model,'r') as f:
        cmodel = f.read()
    
    cmodel = cmodel.replace('\t', '')
    cmodel = cmodel.replace('\"', '')
    cmodel = cmodel.replace(' ', '')
    cmodel = cmodel.lower()
    params = cmodel.split('layer')
    if 'input' not in params[0]:
        params[0] = 'input:data'
    return params





def get_layer_param(params):
    layer_param = {'bottom': []}
    params = params.splitlines()
    for param in params:    
        if ':' in param:
            key = param.split(':')[0]
            value = param.split(':')[1]
            if key=='bottom':
                layer_param['bottom'].append(value)

            elif not key in layer_param.keys():
                layer_param[key] = int(value) if value.isdigit() else value
    return layer_param





def get_input_param(params):
    input_param = {'input_dim':[], 'type':'input', 'name':'data'}
    params = params.splitlines()
    for param in params:    
        if ':' in param:
            key = param.split(':')[0]
            value = param.split(':')[1]
            if key=='input_dim':
                input_param[key].append(int(value))
    return input_param





def get_output_param(params):
    layer_param = get_layer_param(params)
    layer_param['name'] = 'output'
    return layer_param





def cmodel2k_param(cmodel):
    model_params = model_preprocessing(cmodel)
    k_params = []
#     print(len(model_params))
    for i, params in enumerate(model_params):
#         print(i)
        if 'input' in params:
            k_params.append(get_input_param(params))
        elif 'softmax' in params:
#             print(i)
            k_params.append(get_output_param(params))
#             print(get_output_param(params))
        elif 'type' in params:
            k_params.append(get_layer_param(params))
    return k_params





def cmodel2kmodel(cmodel, verbose=False):
    k_params = cmodel2k_param(cmodel)
    k_model = k_param2k_model(k_params, verbose)
    return k_model
    




def k_param2k_model(k_params, verbose=False):
    layers = {}
    debug_layers = {'layer_connect': {}, 'skip_layer': []}
    inputs = None
    outputs= None
    print('building model')
    for layer_params in tqdm(k_params):       
        layer = get_layer(layer_params, layers, verbose, debug_layers)
        if layer!=None:
            if layer_params['type'] == 'input':
                inputs = layer
            elif layer_params['name'] == 'output':
                outputs = layer

# Show layers which don't connect to a next layer
    if verbose:
        print('\nSome layer don\'t connect with a following layer, relu, batch_norm, activation\
output and dropout layers are resonable\n', \
            [key for key, value in debug_layers['layer_connect'].items() if value==0])
        print('\n{} layer has been skipped:\n{}'.format(len(debug_layers['skip_layer']), debug_layers['skip_layer']))

    model = Model(inputs, outputs)
    print('Model builded')
    return model
            






def cweights2kweights(caffe_model, caffe_weights):
    net = caffe.Net(caffe_model,
                   caffe_weights,
                   caffe.TEST)
    print('----------------------------getting weights------------------')
    weights = {}
    for item in net.params.items():    
        name, layer = item
        weights[name] = []
        for w in net.params[name]:
            weights[name].append(w.data)
    return weights
    

def set_bn(name, W, weights):
    scale_name = 'scale' + re.findall(r"bn(.+)", name)[0]
#     print('scale name:{}'.format(scale_name))
    scale_weights = weights[scale_name]
#     print(scale_weights[0].shape)
    W[2] = scale_weights[0]
    W[3] = scale_weights[1]
    return W


def set_weights(model, weights, verbose=False):
    print('setting weights')
    failed = []
    scale_weights = False
    for layer in tqdm(model.layers):
        name = layer.name
        W = layer.get_weights()
        if name in weights.keys():
            for i,w in enumerate(weights[name]):
                if i==0 and len(w.shape)==4:
#                     print(name)
                    w = w.transpose(2, 3, 1, 0)
                if w.shape==W[i].shape:
                    W[i] = w
                else:
                    if 'bn' in name:
                        try:
#                             print('set scale\n')
                            set_bn(name, W, weights)
                            scale_weights = True
                        except:
                            if verbose:
                                print('try to get beta and gamma from scale layer but failed')
                                failed.append(name)            
                                if verbose:
                                    print('error: weights of layer {} not compatible, {} is needed, but given {}'.format(name,W[i].shape,weights[name][i].shape))
                                    W_shape = []
                                    w_shape = []
                                    for x in W:
                                        W_shape.append(x.shape)
                                    for x in weights[name]:
                                        w_shape.append(x.shape)
                                    print('W: {}\ngiven_shape: {}'.format(W_shape, w_shape))
                    # elif 'fc' in name:
                    #     w = w.transpose(1, 0)
                    #     W[i] = w
                    else:
                        failed.append(name)            
                        if verbose:
                            print('error: weights of layer {} not compatible, {} is needed, but given {}'.format(name,W[i].shape,weights[name][i].shape))
                            W_shape = []
                            w_shape = []
                            for x in W:
                                W_shape.append(x.shape)
                            for x in weights[name]:
                                w_shape.append(x.shape)
                            print('W: {}\ngiven_shape: {}'.format(W_shape, w_shape))
        try:
            layer.set_weights(W)
        except:
            failed.append(name)
            print('error: weights of layer {} is in right shape but failed setting'.format(name))
    print('Weights setted, {} layers failed to load the weights\n'.format(len(failed)))
    for i in  range(len(failed)):
        print('{} layers failed to load the weights\n'.format(failed[i]))
    return model




def get_keras_model_with_weights(cmodel, cweights=None, path=None, verbose=False):

    model = cmodel2kmodel(cmodel, verbose)
    
    if cweights!=None:
        weights = cweights2kweights(cmodel, cweights)
        print('----------------------------getting weights2------------------')
        model = set_weights(model, weights, verbose)
    if path!=None:
        print('saving model')
        model.save(path)
        print('model saved')




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model prototxt path .prototxt")
    parser.add_argument("--weights", help="caffe model weights path .caffemodel")
    parser.add_argument("--output", help="output path")
    parser.add_argument("--log", help="show some error info while converting layers and weights")
    args = parser.parse_args()
    if args.model==None:
        print('Error: You must specify a model path use --mode path_to_model')
        exit()
    if args.output==None:
        print('Error: You must specify a output path use --output save_path')
        exit()
    if args.weights==None:
        print('Warn: no weights file specify, will you generate model with out weights?')
        raw_inputs = input('y(es) or n(o) ?\n')
        if not raw_inputs in ['yes', 'y', 'Y', 'YES']:
            exit()
    get_keras_model_with_weights(args.model, cweights=args.weights, path=args.output, verbose=args.log)





