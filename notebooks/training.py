import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import tensorflow as tf
import cryovesnet
# from keras import backend as K
# K.set_image_data_format('channels_last')

# import unetmic.unetmic as umic
import cryovesnet.unetmic.unetmic.unetmic as umic

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# for bigger patch size 64^3
# specifiy path to training data and to folder where to save training weights


save_folder = '/mnt/data/Amin/Data/training_logs/'

# specify number of training data, validation data, batch size
# window_size = 32
# numtot = 900
# numvalid = 200
# batchsize = 50
# folder = '/mnt/data/Amin/Data/train_dataset_32_synaptasome'

## CryoVesNet Training ##
#
# window_size = 64
# numtot = 1600
# numvalid = 250
# batchsize = 25  # 2 gpus
# folder = '/mnt/data/Amin/Data/train_dataset_64_raw_neuron'
#
window_size = 64
numtot = 144
numvalid = 40
batchsize = 25
folder = '/mnt/data/Amin/Data/train_dataset_64_synaptasome_8000'


umic.run_training_multiGPU(save_folder=save_folder, data_folder=folder, num_total=numtot, batch_size=batchsize,
                           num_valid=numvalid, window_size=window_size)



# folder = '/media/amin/mtwo/train_dataset_2d_128_synaptasome_1000/'
# save_folder = '/media/amin/mtwo/train2d/'
#
# #specify number of training data, validation data, batch size
# numtot = 4062
# numvalid = 475
# batchsize = 50
# network_name= "resnet"
#
# umic.run_training_multiGPU_2d(save_folder = save_folder, data_folder = folder,
#                               num_total = numtot,batch_size = batchsize, num_valid = numvalid,network_name=network_name)