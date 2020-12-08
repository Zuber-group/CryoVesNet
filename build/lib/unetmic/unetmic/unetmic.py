import os
import numpy as np
from tqdm import tqdm
import skimage.io
import mrcfile

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv3D

#from keras import backend as K
#from keras.models import Model
#from keras.callbacks import ModelCheckpoint
#from keras.layers import Input, Conv3D

from . import blocks

def train_generator(folder, numtot, batchsize):
    rand_arr = np.random.choice(numtot, size=numtot, replace = False)
    num = 0
    while num > -1:
        
        batch_counter = 0
        
        im_batch = []
        mask_batch = []
        for i in range(batchsize):
            im_batch.append(np.load(folder+'/image_'+str(rand_arr[num])+'.npy')[:,:,:,np.newaxis])
            mask_batch.append(np.load(folder+'/mask_'+str(rand_arr[num])+'.npy')[:,:,:,np.newaxis])
            num += 1
            if num==numtot:
                num=0
                rand_arr = np.random.choice(numtot, size=numtot, replace = False)
        
        im_batch = (im_batch-np.mean(im_batch))/np.std(im_batch)
        yield (np.array(im_batch).astype(np.float32),np.array(mask_batch).astype(np.float32))

        
def valid_generator(folder, numvalid, numtot, batchsize):
    num = 0
    while num > -1:
        
        batch_counter = 0
        
        im_batch = []
        mask_batch = []
        for i in range(batchsize):
            im_batch.append(np.load(folder+'/image_'+str(num+numtot)+'.npy')[:,:,:,np.newaxis])
            mask_batch.append(np.load(folder+'/mask_'+str(num+numtot)+'.npy')[:,:,:,np.newaxis])
            num += 1
            if num==numvalid:
                num=0
        
        im_batch = (im_batch-np.mean(im_batch))/np.std(im_batch)
        yield (np.array(im_batch).astype(np.float32),np.array(mask_batch).astype(np.float32))
        
        
        
def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def create_unet_3d(inputsize = (32,32,32,1)):
    
    n_depth = 2
    n_filter_base = 16
    kernel_size = (3,3,3)
    activation= 'relu'
    batch_norm=False
    dropout=0.0
    n_conv_per_depth = 2
    pool_size=(2,2,2)
    n_channel_out = 1
    n_dim = len(kernel_size)

    input = Input(inputsize, name = "input")
    
    unet = blocks.unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)
    
    final = Conv3D(n_channel_out, (1,)*n_dim, activation='sigmoid')(unet)
    
    unet3d = Model(inputs=input, outputs=final)
    unet3d.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    
    return unet3d


def run_training(network, save_folder, folder, numtot, batchsize, numvalid):
    
    model_checkpoint = ModelCheckpoint(save_folder+'weights.h5', monitor='val_loss', save_best_only=True)
    #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        
    network.fit_generator(generator=train_generator(folder, numtot, batchsize),
                          validation_data = valid_generator(folder, numvalid, numtot, batchsize),
                          validation_steps = np.floor(numvalid/batchsize),
                          steps_per_epoch= np.floor(numtot/batchsize),
                          epochs=200, verbose=1,callbacks=[model_checkpoint],class_weight=[1,10])
    
def run_segmentation(image, unet3d):
    #region of interest. Middle image region kept after segmentation
    #the network input needs to be larger than this (e.g. 64) to avoid
    #tiling effects
    
    network_size = unet3d.input.get_shape().as_list()[1]
    roi = 24
    wsize = network_size
    to_pad0 = roi-(np.array(image.shape)%roi)
    to_pad = int((wsize-roi)/2)
    
    #pad image to avoid edge effects
    image_pd = np.pad(image,[[0,x] for x in roi-(np.array(image.shape)%roi)],mode='constant', constant_values = 0)
    ideal_size = np.array(image_pd.shape)
    image_pd = np.pad(image_pd,to_pad,mode='constant', constant_values = 0)
    
    #run the segmentation by tiling the image
    image_mask = np.zeros(ideal_size)
    numit = 0
    for i in tqdm(range(0,image.shape[0],roi)):
        #print(i)
        for j in range(0,image.shape[1],roi):
            #print(j)
            for k in range(0,image.shape[2],roi):
                #print(k)
                numit+=1
                test = unet3d.predict(image_pd[i:i+wsize, j:j+wsize, k:k+wsize][np.newaxis,:,:,:,np.newaxis])
                test_im = test[0,:,:,:,0]
                image_mask[i:i+roi,j:j+roi,k:k+roi]\
                = test_im[to_pad:to_pad+roi,to_pad:to_pad+roi,to_pad:to_pad+roi]
    image_mask = image_mask[0:image_mask.shape[0]-to_pad0[0],0:image_mask.shape[1]-to_pad0[1],0:image_mask.shape[2]-to_pad0[2]]

    return image_mask

def load_raw(path_to_file):
    if os.path.splitext(os.path.split(path_to_file)[1])[1] == '.tif':
        image = skimage.io.imread(path_to_file)
    
    else:
        image = mrcfile.open(path_to_file)
        image = image.data
    
    return image

def save_seg_output(image_mask, path_to_file, folder_to_save):
    #check if folder exists
    if not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)

    #save the resulting mask as npy file
    np.save(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_segUnet.npy',image_mask)
    #create an 8bit image that one can visualize in Fiji and save it
    #image_mask_int = (255*np.flip(image_mask,axis = 1)).astype(np.uint8)
    image_mask_int = (255*image_mask).astype(np.uint8)
    skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_segUnet.tiff',image_mask_int, plugin = 'tifffile')


    
    