import os 
import numpy as np
import skimage
import skimage.transform
import skimage.morphology
import skimage.io
import pandas as pd
import mrcfile

from . import unetmic as umic
from . import segment as segment

def find_threshold(image, image_mask):
    
    #first, calculate shell-intensity at different thresholds
    shell_pixels = []
    for th in np.arange(0.8, 1, 0.01):
        image_mask_bin = np.zeros(image_mask.shape)
        image_mask_bin[image_mask>th] = 1

        image_mask_bin_erode = skimage.morphology.binary_erosion(image_mask_bin, selem=skimage.morphology.ball(1))
        image_shell = image_mask_bin - image_mask_bin_erode

        shell_pixels.append(image[image_shell.astype(bool)])

    mean_shell_val = [np.mean(x) for x in shell_pixels]
    #find optimal threshold
    opt_th = np.arange(0.8, 1, 0.01)[np.argmin(mean_shell_val)]
    
    return opt_th, mean_shell_val


def mask_clean_up(image_label):
    measures = skimage.measure.regionprops_table(image_label, properties=('filled_area','label','extent'))
    measures_pd = pd.DataFrame(measures)
    #select objects by extent
    sel_labels = measures_pd[(measures_pd.extent>0.3)&(measures_pd.extent<0.7)].label.values
    #use array indexing to only keep "good" objects"
    indices = np.array([i if i in sel_labels else 0 for i in np.arange(measures_pd.label.max()+1)])
    image_label_clean = indices[image_label]
    #relabel
    clean_mask = image_label_clean>0
    clean_labels = skimage.morphology.label(clean_mask)
    
    return clean_mask, clean_labels

def full_segmentation(network_size, unet_weigth_path, path_to_file, folder_to_save, rescale = 1, gauss = False):
    #load the network and weights
    unet_vesicle = umic.create_unet_3d(inputsize=(network_size,network_size,network_size,1))
    unet_vesicle.load_weights(unet_weigth_path)

    #load image
    image = umic.load_raw(path_to_file)
    #rescale
    if rescale !=1:
        image = skimage.transform.rescale(image, scale=rescale, preserve_range = True).astype(np.int16)
    if gauss:
        image = skimage.filters.gaussian(image, preserve_range = True).astype(np.int16)
    
    if (rescale ==1) or gauss:
        skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_processed.tiff',
                      image, plugin = 'tifffile')
    #normalize image
    image = (image-np.mean(image))/np.std(image)
    #do training
    image_mask = umic.run_segmentation(image, unet_vesicle)
    #save raw output as npy
    umic.save_seg_output(image_mask, path_to_file, folder_to_save)

    opt_th, _ = find_threshold(image, image_mask)

    image_label_opt = skimage.morphology.label(image_mask>opt_th)

    clean_mask, clean_labels = mask_clean_up(image_label_opt)

    #export 
    #clean_mask = (255*np.flip(clean_mask,axis = 1)).astype(np.uint8)
    clean_mask = (255*clean_mask).astype(np.uint8)
    skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_mask.tiff',
                      clean_mask, plugin = 'tifffile')

    clean_labels =clean_labels.astype(np.uint16)
    skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_label.tiff',
                      clean_labels, plugin = 'tifffile')
    
    skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_wreal_mask.tiff',
                      image_mask, plugin = 'tifffile')


    
    with mrcfile.new(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_label.mrc') as mrc:
        mrc.set_data(clean_labels+10)
