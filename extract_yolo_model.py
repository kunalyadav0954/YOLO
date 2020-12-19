# Script for reading yolo_v2 config file for darknet 19 architecture
# Creates Keras model and loads it weights specified by the .weights file
# The generated Keras model is stored in a .h5 file
import argparse
import configparser
import io
import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Lambda,
                          MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot


def reorg_block(x):
    """
    Function to be fed into keras Lambda to reorganize input into 4(block size=2) blocks and concatenate them
    :param x: the input tensor
    :return: the reorganized block
    """
    import tensorflow as tf

    return tf.space_to_depth(x, block_size=2)

def reorg_shape(input_shape):
    """
    Function to be fed into keras Lambda to define the output shape of the reorganized block
    :param input_shape: the shape of the tensor that is being reorganized
    :return: the output_shape of reorganized block
    """
    if input_shape[1]:
        output_shape = (input_shape[0],input_shape[1]//2,input_shape[2]//2,input_shape[3]*4)
    else:
        output_shape = (input_shape[0], None, None, input_shape[3] * 4)

    return output_shape


# Creating a parser object to parse arguments that will be passed to the script through command line
parser = argparse.ArgumentParser(description="Model extractor for darknet-19 architecture type")
parser.add_argument('config_path', help='path to Darknet config file')
parser.add_argument('weights_path', help='path to Darknet weights file')
parser.add_argument('output_path', help='path to out put keras model file in .h5 format')
parser.add_argument('-p', '--plot_model', help='plot generated keras model and save as image',
                    action='store_true')


def config_to_file(config_file):
    """
    Converts a config file into a file type object with unique section names ex:- conv_1, conv_2, etc
    :param config_file: path to the config file
    :return: output_stream - a file type object with unique section anmes which will further be parsed
                             be parsed by configparser into a dictionary
    """

    section_count = defaultdict(int) # dictionary of int type for counting section number
    output_stream = io.StringIO() # a file type object
                                  # will be used for storing the info for configparser
    with open(config_file) as file:
        for line in  file:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' +str(section_count[section])
                section_count[section] += 1
                line= line.replace(section , _section)
            output_stream.write(line)
    output_stream.seek(0)  # move counter to start of file

    return output_stream

def _main(args):

    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    output_path = os.path.expanduser(args.output_path)

    output_root = os.path.splitext(output_path)[0]

    unique_file = config_to_file(config_path)
    cfg_parser = configparser.ConfigParser()    # creating config parser object
    cfg_parser.read_file(unique_file)           # reading unique_file so that it is parsed into a dictionary

    print('Loading weights.....\n')
    weights_file= open(weights_path, 'rb')      # reading .weights file in binary format
    weights_header= np.ndarray(shape=(4,), dtype='int32' , buffer=weights_file.read(16))
    print('weights_header = {}'.format(weights_header))

    print('\nParsing config file....\n')

    #Creating input place holder for the keras model
    image_height = int(cfg_parser['net_0']['height'])
    image_width = int( cfg_parser['net_0']['width'] )
    weight_decay = float( cfg_parser['net_0']['decay'] )
    prev_layer = Input(shape= (image_height, image_width , 3))
    all_layers = [prev_layer]        # This list will be used for storing the outputs of all layers

    print('The Models input dimension is : ({},{},3)'.format(image_height,image_width))
    print('Regularization constant for the model, lambda : {}'.format(weight_decay))

    count=0   # for counting total number of parameters
    l=0       #for layer numbers
    for section in cfg_parser.sections():
        if section!= 'region':
            print('\n-------------LAYER {}----------------'.format(l))
        print('\nParsing section : {}'.format(section))

        if section.startswith('convolutional'):
            filters =int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding ='same' if pad==1 else 'valid'

            # In darknet weights file, the weights are stored as:
            # [bias/beta ,[gamma, mean ,variance] , conv weights]
            prev_layer_shape = K.int_shape(prev_layer)
            weights_shape = (size,size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters , weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            # Extracting weights for bias/beta (used as beta if batch normalize is true else used as bias weights)
            conv_bias = np.ndarray(shape=(filters,) , dtype='float32' , buffer=weights_file.read(filters*4))
            count += filters

            # Extracting weights(gamma,mean,varinace) for each filter if batch normalize is true
            if batch_normalize:
                bn_weights = np.ndarray(shape=(3,filters) , dtype='float32' , buffer=weights_file.read(filters*12))
                count += filters*3
                bn_weight_list =[bn_weights[0],conv_bias,bn_weights[1],bn_weights[2]]  # [gamma,beta,mean,variance]
                                                                                        # for each filter
            conv_weights = np.ndarray(shape=darknet_w_shape,dtype='float32',
                                      buffer=weights_file.read(weights_size*4) )
            count += weights_size

            #converting weights shape from darknet to Keras convention
            conv_weights = np.transpose(conv_weights, [2,3,1,0])
            if batch_normalize:
                conv_weights = [conv_weights]
            else:
                conv_weights=[conv_weights,conv_bias]

            # Creating conv2D layer
            conv_layer = (Conv2D(filters,(size,size),strides=(stride,stride),padding=padding,
                                kernel_regularizer=l2(weight_decay),use_bias=not batch_normalize,
                                weights=conv_weights,activation=None))(prev_layer)
            if batch_normalize:
                conv_layer=BatchNormalization(weights=bn_weight_list)(conv_layer)

            prev_layer=conv_layer

            if activation=='linear':
                all_layers.append(prev_layer)
            elif activation=='leaky':
                act_layer= LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer=act_layer
                all_layers.append(prev_layer)

            output_layer_shape =K.int_shape(prev_layer)

            print('no. of filters = {}\nsize = {}\nstride = {}\npadding = {}'.format(filters,size,
                                                                                     stride,padding))
            print('batch normalize = {}\nactivation = {}'.format(batch_normalize,activation))
            print('\nInput tensor shape = {}\nOutput tensor shape = {}\n'.format(prev_layer_shape,
                                                                                 output_layer_shape))

        elif section.startswith('maxpool'):
            prev_layer_shape = K.int_shape(prev_layer)
            size= int( cfg_parser[section]['size'] )
            stride= int( cfg_parser[section]['stride'] )
            all_layers.append( MaxPooling2D(pool_size=(size,size),strides=(stride,stride),
                                            padding='same')(prev_layer) )
            prev_layer = all_layers[-1]
            output_layer_shape =K.int_shape(prev_layer)

            print('size = {}\nstride = {}'.format(size,stride))
            print('\nInput tensor shape = {}\nOutput tensor shape = {}\n'.format(prev_layer_shape,
                                                                                 output_layer_shape))

        elif section.startswith('avgpool'):
            if cfg_parser.items(section) != []:
                raise ValueError('{} with params unsupported.'.format(section))
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('route'):
            # creating a list consisting of layer numbers to route
            layer_numbers=[int(i) for i in cfg_parser[section]['layers'].split(',')]
            # Fetching corresponding layers
            layers= [all_layers[i] for i in layer_numbers]
            if len(layer_numbers) > 1:
                print('Concatenating following layers : ')
                j=0
                for i in layer_numbers:
                    print('layer{} ,shape = {}'.format(len(all_layers)+i,K.int_shape(layers[j])))
                    j+=1
                concatenated_layer = concatenate(layers)
                prev_layer=concatenated_layer
                all_layers.append(prev_layer)
            else:
                print('Creating a skip connection from layer{}'.format(len(all_layers)+layer_numbers[0]))
                skip_layer = layers[0]
                prev_layer = skip_layer
                all_layers.append(prev_layer)

            output_layer_shape = K.int_shape(prev_layer)
            print('\nOutput tensor shape = {}\n'.format(output_layer_shape))

        elif section.startswith('reorg'):
            print('\nReorganizing input into 4 small blocks and concatenating them.....')
            input_shape=K.int_shape(prev_layer)
            all_layers.append( Lambda(function=reorg_block, output_shape=reorg_shape, name='reorg_x4')(prev_layer) )
            prev_layer=all_layers[-1]
            output_shape= K.int_shape(prev_layer)
            print('Reorganized input of shape {} into output of shape {}'.format(input_shape, output_shape))

        elif section.startswith('region'):
            with open('{}_anchors.txt'.format(output_root),'w') as f:
                print(cfg_parser[section]['anchors'], file=f)

            print('\nSaved anchors in the same directory as model')


        l+=1  # Increment the layer number to be displayed

    print('\n-------------------------------------------------------\n')
    # Creating and saving model
    model= Model(inputs=all_layers[0], outputs=all_layers[-1])
    model.save('{}'.format(output_path))
    print(model.summary())
    remaining_weights=len(weights_file.read()) // 4
    weights_file.close()
    print('saved model successfully')
    print('Read {} out of {} weights'.format(count,count+remaining_weights))
    if remaining_weights>0:
        print('WARNING : Unused weights')
    if args.plot_model:
        plot(model,to_file='{}.png'.format(output_root), show_shapes=True)
        print('model plot saved in the same directory as model')


if __name__ == '__main__':  # True when the script is run through command line and not imported as module
    _main(parser.parse_args())
