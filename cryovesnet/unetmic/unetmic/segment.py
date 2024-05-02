import os
import numpy as np
import skimage
import skimage.transform
import skimage.morphology
import skimage.io
import pandas as pd
import mrcfile

from tqdm import tqdm

from . import unetmic as umic
from . import segment as segment
from tensorflow.keras.models import load_model


def find_threshold(image, image_mask,min_th=0.8,max_th=1,step=0.01):
    # first, calculate shell-intensity at different thresholds
    shell_pixels = []
    for th in tqdm(np.arange(min_th, max_th, step), desc='finding global threshold on unet mask'):
        image_mask_bin = np.zeros(image_mask.shape)
        image_mask_bin[image_mask > th] = 1

        image_mask_bin_erode = skimage.morphology.binary_erosion(image_mask_bin, footprint=skimage.morphology.ball(1))
        image_shell = image_mask_bin - image_mask_bin_erode

        shell_pixels.append(image[image_shell.astype(bool)])

    mean_shell_val = [np.mean(x) for x in shell_pixels]
    # find optimal threshold
    opt_th = np.arange(min_th, max_th, step)[np.argmin(mean_shell_val)]

    return opt_th, mean_shell_val


def mask_clean_up(image_label, background=0):
    """
    cleans image_label. sets the removed labels to background. Reorder labels not to introduce gaps in indices.
    :param image_label: array that contains the labels
    :param background: value to which the excluded labels are set.
    :return: cleaned label array
    """
    print("Clean Mask")
    # BZ removed from properties "filled_area" which was not used.
    measures = skimage.measure.regionprops_table(image_label, properties=('label', 'extent'))
    measures_pd = pd.DataFrame(measures)
    # select objects by extent
    all_labels = measures_pd.label.values
    sel_labels = measures_pd[(measures_pd.extent > 0.3) & (measures_pd.extent < 0.7)].label.values
    unselected_labels = sorted(set(all_labels) - set(sel_labels))
    # BZ made the next lines and loop for reordering labels in a more efficient way
    selection_mask = np.isin(image_label, sel_labels, invert=True)
    image_label[selection_mask] = background
    for l in tqdm(unselected_labels[::-1], desc='reorganizing labels after label clean up'):
        image_label = np.where(image_label > l, image_label - 1, image_label)
    clean_mask = image_label > background
    return clean_mask, image_label


def full_segmentation(unet_weigth_path, path_to_file, folder_to_save, rescale=1, gauss=False, input_size_multiplier=2):
    # load the network and weights
    # network_size=32
    # unet_vesicle = umic.create_unet_3d(inputsize=(network_size, network_size, network_size, 1), n_depth=2,
    #                                    n_filter_base=16, batch_norm=True, dropout=0.0, n_conv_per_depth=2)
    # # unet_vesicle = umic.create_unet_3d(inputsize=(network_size,network_size,network_size,1),n_depth = 3, n_filter_base = 32, batch_norm = True, dropout = 0.0, n_conv_per_depth = 2)
    # unet_vesicle.load_weights(unet_weigth_path)

    unet_vesicle = load_model(unet_weigth_path, custom_objects={
        'weighted_binary_crossentropy': umic.weighted_binary_crossentropy,
        'dice_coef': umic.dice_coef
    })
    print(unet_vesicle.input.shape)
    print(len(unet_vesicle.input.shape))
    # check if 3D network or 2D network
    if (len(unet_vesicle.input.shape)>4):

        train_network_size = unet_vesicle.input.shape[1].value
        n_filter_base = {32: 16, 64: 32}[train_network_size]
        n_depth = {32: 2, 64: 3}[train_network_size]
        inference_input_size = train_network_size * input_size_multiplier
        unet_vesicle = umic.create_unet_3d(inputsize=(*([inference_input_size] * 3), 1), n_depth=n_depth,
                                           n_filter_base=n_filter_base, batch_norm=True, dropout=0.0, n_conv_per_depth=2)
        unet_vesicle.load_weights(unet_weigth_path)
    else:

        unet_vesicle.load_weights(unet_weigth_path)
    # load image
    image = umic.load_raw(path_to_file)
    # rescale
    if rescale != 1:
        image = skimage.transform.rescale(image, scale=rescale, preserve_range=True).astype(np.int16)
    if gauss:
        image = skimage.filters.gaussian(image, preserve_range=True).astype(np.int16)

    if (rescale != 1) or gauss:
        skimage.io.imsave(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[
            0] + '_processed.tiff',
                          image, plugin='tifffile')
    # normalize image
    image = (image - np.mean(image)) / np.std(image)
    # do training
    if (len(unet_vesicle.input.shape) > 4):

        image_mask = umic.run_segmentation(image, unet_vesicle)
        # save raw output as npy
        umic.save_seg_output(image_mask, path_to_file, folder_to_save)
    else:
        image_mask = umic.run_segmentation_2d(image, unet_vesicle)
        # save raw output as npy
        umic.save_seg_output(image_mask, path_to_file, folder_to_save)

    # opt_th, _ = find_threshold(image, image_mask)

    # image_label_opt = skimage.morphology.label(image_mask>opt_th)

    # clean_mask, clean_labels = mask_clean_up(image_label_opt)

    # export
    # clean_mask = (255*np.flip(clean_mask,axis = 1)).astype(np.uint8)
    # clean_mask = (255*clean_mask).astype(np.uint8)
    # skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_mask.tiff',
    #                   clean_mask, plugin = 'tifffile')
    #
    # clean_labels =clean_labels.astype(np.uint16)
    # skimage.io.imsave(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_label.tiff',
    #                   clean_labels, plugin = 'tifffile')

    skimage.io.imsave(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[
        0] + '_wreal_mask.tiff',
                      image_mask, plugin='tifffile')

    # with mrcfile.new(os.path.normpath(folder_to_save)+'/'+os.path.splitext(os.path.split(path_to_file)[1])[0]+'_clean_label.mrc') as mrc:
    #     mrc.set_data(clean_labels+10)
