import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import numpy as np
import matplotlib.pyplot as plt
import os
import mrcfile
import pdb
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import unetmic.unetmic as umic
import unetmic.segment as segseg
from tqdm import tqdm





def min_volume_of_vesicle(path_to_file):
    # read voxel size of tomogram
    with mrcfile.open(path_to_file) as tomo:
        voxel_size = float(tomo.voxel_size.x)
        # print(voxel_size)
    radius_thr = 12
    volume_of_vesicle = (4.0 / 3.0) * np.pi * (radius_thr / voxel_size * 10) ** 3
    return volume_of_vesicle


def save_label_to_mrc(labels,path_to_file,folder_to_save,suffix):

    print(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.mrc')
    with mrcfile.new(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.mrc', overwrite=True) as mrc:
        mrc.set_data(labels)
    return os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.mrc'




def save_label_to_tiff(labels,path_to_file,folder_to_save,suffix):
    skimage.io.imsave(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.tiff',
                      labels, plugin='tifffile')
    return os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.tiff'



# here we use mask_image (probabilty map) comes from our network
# for each vesicles we compute theroshold from this map
# if you set dilation number you will have more dilated vesicles
# if you set convex as 1 then all the pacman gona die :D

def find_threshold_per_vesicle_intensity(int_image, image_label, mask_image, dilation, convex,volume_of_vesicle):
    '''Optimize the size of each vesicle mask to minimize shell intensity'''

    # calculate the properties of the labelled regions
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('centroid', 'label', 'bbox'))
    mean_shell_arg = []
    # print(int_image.shape,type(int_image.shape),np.shape(int_image))
    win_size = 20
    win_size= int((int_image.shape[2]/512)*10)
    print(int_image.shape,win_size)
    corrected_labels = np.zeros(int_image.shape)
    for i in range(1, len(vesicle_regions['label'])):
        # cut out regions
        #         print(np.shape(int_image))
        sub_int_im = int_image[vesicle_regions['bbox-0'][i] - win_size:vesicle_regions['bbox-3'][i] + win_size + 1,
                     vesicle_regions['bbox-1'][i] - win_size:vesicle_regions['bbox-4'][i] + win_size + 1,
                     vesicle_regions['bbox-2'][i] - win_size:vesicle_regions['bbox-5'][i] + win_size + 1]

        sub_mask = mask_image[vesicle_regions['bbox-0'][i] - win_size:vesicle_regions['bbox-3'][i] + win_size + 1,
                   vesicle_regions['bbox-1'][i] - win_size:vesicle_regions['bbox-4'][i] + win_size + 1,
                   vesicle_regions['bbox-2'][i] - win_size:vesicle_regions['bbox-5'][i] + win_size + 1]

        old_label = image_label[vesicle_regions['bbox-0'][i] - win_size:vesicle_regions['bbox-3'][i] + win_size + 1,
                    vesicle_regions['bbox-1'][i] - win_size:vesicle_regions['bbox-4'][i] + win_size + 1,
                    vesicle_regions['bbox-2'][i] - win_size:vesicle_regions['bbox-5'][i] + win_size + 1]
        old_label = old_label == vesicle_regions['label'][i]

        new_opt_thr, new_mean_shell_val = my_threshold(sub_int_im, sub_mask)

        best_mask = ((((((sub_mask > new_opt_thr))))))
        omit = 0
        if all(np.array(np.shape(sub_mask)) != 0):
            image_label_opt = skimage.morphology.label(sub_mask >= new_opt_thr)
            sub_label = image_label_opt

            p = int(abs(vesicle_regions['bbox-3'][i] - vesicle_regions['bbox-0'][i]) / 2) + win_size
            q = int(abs(vesicle_regions['bbox-4'][i] - vesicle_regions['bbox-1'][i]) / 2) + win_size
            r = int(abs(vesicle_regions['bbox-5'][i] - vesicle_regions['bbox-2'][i]) / 2) + win_size
            if (image_label_opt[p, q, r] == 0):
                a, b, c = ndimage.measurements.center_of_mass(image_label_opt, vesicle_regions['label'][i])
                if not any(np.isnan([a, b, c])):
                    sub_label = image_label_opt
                    a = int(a)
                    b = int(b)
                    c = int(c)

                    new_lable_of_ves = image_label_opt[a, b, c]
                    if new_lable_of_ves != 0:
                        sub_label = sub_label == new_lable_of_ves
                        best_mask = sub_label
                    else:
                        print('out')
                        best_mask = old_label.copy()
                else:
                    print('bad vesicles')
                    best_mask = old_label.copy()
                    omit = 1
            else:
                sub_label = sub_label == image_label_opt[p, q, r]
                best_mask = sub_label
        else:
            print('bad vesicles', i)
            best_mask = old_label.copy()
            omit = 1

        a = (best_mask.reshape([-1, 1])).transpose()
        if omit == 0 and len(a[a > 0]) < volume_of_vesicle:
            print('small')
            omit = 1

        best_param, best_mask = adjust_shell_intensity(sub_int_im, best_mask)

        mean_shell_arg.append(new_mean_shell_val)
        # bring pacman above OUT!
        # if convex == 1:
        #     #             best_mask=skimage.morphology.convex_hull_image(best_mask, offset_coordinates=False)
        #     try:
        #         best_mask = skimage.morphology.convex_hull_image(best_mask, offset_coordinates=False)
        #     except:
        #         print("?", vesicle_regions['label'][i])
        if omit == 0:
            px, py, pz = np.where(best_mask)
            corrected_labels[px + vesicle_regions['bbox-0'][i] - win_size,
                             py + vesicle_regions['bbox-1'][i] - win_size,
                             pz + vesicle_regions['bbox-2'][i] - win_size] = i + 1

    #     mask_adjusted = skimage.morphology.binary_erosion(corrected_labels>=1, selem=skimage.morphology.ball(1))
    #     mask_adjusted = skimage.morphology.binary_dilation(corrected_labels>=1, selem=skimage.morphology.ball(1))
    #     corrected_labels = skimage.morphology.label(mask_adjusted)
    if dilation == 1:
        corrected_labels = skimage.morphology.binary_dilation(corrected_labels >= 1, selem=skimage.morphology.ball(1))

    corrected_labels = skimage.morphology.label(corrected_labels >= 1, connectivity=1)
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels, mean_shell_arg


def pacman_killer(image_label):
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('centroid', 'label', 'bbox'))
    mean_shell_arg = []
    # print(int_image.shape,type(int_image.shape),np.shape(int_image))
    win_size = 1
    win_size = int((image_label.shape[2] / 512) * 10)
    print(image_label.shape, win_size)
    corrected_labels = np.zeros(image_label.shape)
    for i in tqdm(range(1, len(vesicle_regions['label']))):

        old_label = image_label[vesicle_regions['bbox-0'][i] - win_size:vesicle_regions['bbox-3'][i] + win_size + 1,
                    vesicle_regions['bbox-1'][i] - win_size:vesicle_regions['bbox-4'][i] + win_size + 1,
                    vesicle_regions['bbox-2'][i] - win_size:vesicle_regions['bbox-5'][i] + win_size + 1]
        old_label = old_label == vesicle_regions['label'][i]


        #             best_mask=skimage.morphology.convex_hull_image(best_mask, offset_coordinates=False)
        try:
            old_label = skimage.morphology.convex_hull_image(old_label, offset_coordinates=False)
        except:
            print("?", vesicle_regions['label'][i])

        px, py, pz = np.where(old_label)
        corrected_labels[px + vesicle_regions['bbox-0'][i] - win_size,
                         py + vesicle_regions['bbox-1'][i] - win_size,
                         pz + vesicle_regions['bbox-2'][i] - win_size] = i + 1
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels





def fast_pacman_killer(image_label):
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('centroid', 'label', 'bbox'))
    mean_shell_arg = []
    # print(int_image.shape,type(int_image.shape),np.shape(int_image))
    win_size = 1
    win_size = int((image_label.shape[2] / 512) * 10)
    print(image_label.shape, win_size)
    corrected_labels = np.zeros(image_label.shape)
    for i in tqdm(range(1, len(vesicle_regions['label']))):

        old_label = image_label[vesicle_regions['bbox-0'][i] - win_size:vesicle_regions['bbox-3'][i] + win_size + 1,
                    vesicle_regions['bbox-1'][i] - win_size:vesicle_regions['bbox-4'][i] + win_size + 1,
                    vesicle_regions['bbox-2'][i] - win_size:vesicle_regions['bbox-5'][i] + win_size + 1]
        old_label = old_label == vesicle_regions['label'][i]


        #             best_mask=skimage.morphology.convex_hull_image(best_mask, offset_coordinates=False)
        old_label = skimage.morphology.convex_hull_image(old_label, offset_coordinates=False)

        px, py, pz = np.where(old_label)
        corrected_labels[px + vesicle_regions['bbox-0'][i] - win_size,
                         py + vesicle_regions['bbox-1'][i] - win_size,
                         pz + vesicle_regions['bbox-2'][i] - win_size] = i + 1
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels


def my_threshold(image, image_mask):
    # first, calculate shell-intensity at different thresholds
    shell_pixels = []
    for th in np.arange(0.8, 1, 0.01):
        image_mask_bin = np.zeros(image_mask.shape)
        image_mask_bin[image_mask > th] = 1

        image_mask_bin_erode = skimage.morphology.binary_erosion(image_mask_bin, selem=skimage.morphology.ball(1))
        image_shell = image_mask_bin - image_mask_bin_erode

        shell_pixels.append(image[image_shell.astype(bool)])

    mean_shell_val = [np.mean(x) for x in shell_pixels]
    # find optimal threshold
    opt_th = np.arange(0.8, 1, 0.01)[np.argmin(mean_shell_val)]

    return opt_th, mean_shell_val


def adjust_shell_intensity(image_int, image_labels):
    '''For a given sub-region adjust the size the vesicle mask to minimize its average shell intensity'''

    mean_shell = []

    # calculate the mean shell value in the original segmentation.
    mask_adjusted = image_labels.copy()
    # create a slightly eroded version of the region
    image_mask_bin_erode = skimage.morphology.binary_erosion(mask_adjusted, selem=skimage.morphology.ball(1))

    # combine original image and eroded one to keep only the shell
    image_shell = mask_adjusted ^ image_mask_bin_erode

    # recover pixels in the shell and calculate their mean intensity
    shell_pixels = image_int[image_shell.astype(bool)]
    mean_shell.append([0, np.mean(shell_pixels)])
    best_param = 0
    best_value = np.mean(shell_pixels)
    best_mask = image_labels.copy()
    for i in range(1, 8):
        mask_adjusted = skimage.morphology.binary_dilation(image_labels, selem=skimage.morphology.ball(i))
        image_mask_bin_erode = skimage.morphology.binary_erosion(mask_adjusted, selem=skimage.morphology.ball(1))
        image_shell = mask_adjusted ^ image_mask_bin_erode
        shell_pixels = image_int[image_shell.astype(bool)]
        meanval = np.mean(shell_pixels)
        mean_shell.append([i, meanval])
        if meanval < best_value:
            best_value = meanval
            best_param = i
            best_mask = mask_adjusted

    return best_param, best_mask  # , mean_shell


# this function give 2 label map OLD and NEW
# and corrct any thing you want not happend in new one which is not desired and turn it back to old one
# here we had intersection problem of vesicles after changing theroshold
# at the end this function OR 2 given input (which is source of some bug = we should bring it to top -> later decion needed in group)
# but give u larger map as result

def oneToOneCorrection(old_label, new_label):
    vesicle_regions = skimage.measure.regionprops_table(old_label, properties=('centroid', 'label', 'bbox'))
    myvesicle_regions = skimage.measure.regionprops_table(new_label, properties=('centroid', 'label', 'bbox'))
    mean_shell_arg = []

    #     corrected_labels = np.zeros(new_label.shape)
    corrected_labels = new_label.copy()

    vesicles = []
    for i in range(1, len(vesicle_regions['label'])):
        sub_old_label = old_label[vesicle_regions['bbox-0'][i] - 10:vesicle_regions['bbox-3'][i] + 11,
                        vesicle_regions['bbox-1'][i] - 10:vesicle_regions['bbox-4'][i] + 11,
                        vesicle_regions['bbox-2'][i] - 10:vesicle_regions['bbox-5'][i] + 11]
        sub_old_label = sub_old_label == vesicle_regions['label'][i]

        sub_new_label = new_label[vesicle_regions['bbox-0'][i] - 10:vesicle_regions['bbox-3'][i] + 11,
                        vesicle_regions['bbox-1'][i] - 10:vesicle_regions['bbox-4'][i] + 11,
                        vesicle_regions['bbox-2'][i] - 10:vesicle_regions['bbox-5'][i] + 11]

        a, b, c = ndimage.measurements.center_of_mass(sub_old_label, vesicle_regions['label'][i])
        if any(np.isnan([a, b, c])):
            vesicles += [0]
        else:

            a = int(a)
            b = int(b)
            c = int(c)
            vesicles += [sub_new_label[a, b, c]]

    #     print(vesicle_regions['label'])
    #     print(myvesicle_regions['label'])
    unique, counts = np.unique(vesicles, return_counts=True)
    intersected = (unique[counts > 1])
    for j in intersected[1:]:
        places = np.where(vesicles == j)
        places = places[0]
        #         print(places)
        #         print(j,myvesicle_regions['label'][j-1])
        sub_temp_label = new_label[myvesicle_regions['bbox-0'][j - 1] - 10:myvesicle_regions['bbox-3'][j - 1] + 11,
                         myvesicle_regions['bbox-1'][j - 1] - 10:myvesicle_regions['bbox-4'][j - 1] + 11,
                         myvesicle_regions['bbox-2'][j - 1] - 10:myvesicle_regions['bbox-5'][j - 1] + 11]
        sub_temp_label = sub_temp_label == j

        px, py, pz = np.where(sub_temp_label)
        corrected_labels[px + myvesicle_regions['bbox-0'][j - 1] - 10,
                         py + myvesicle_regions['bbox-1'][j - 1] - 10,
                         pz + myvesicle_regions['bbox-2'][j - 1] - 10] = 0

        for m in places:
            #             print(m)
            print(j, vesicle_regions['label'][m], m)
            sub_orig_label = old_label[vesicle_regions['bbox-0'][m] - 10:vesicle_regions['bbox-3'][m] + 11,
                             vesicle_regions['bbox-1'][m] - 10:vesicle_regions['bbox-4'][m] + 11,
                             vesicle_regions['bbox-2'][m] - 10:vesicle_regions['bbox-5'][m] + 11]
            sub_orig_label = sub_orig_label == vesicle_regions['label'][m]

            px, py, pz = np.where(sub_orig_label)
            corrected_labels[px + vesicle_regions['bbox-0'][m] - 10,
                             py + vesicle_regions['bbox-1'][m] - 10,
                             pz + vesicle_regions['bbox-2'][m] - 10] = 1000 + m

    best_corrected_labels = (corrected_labels >= 1) | (old_label >= 1)
    best_corrected_labels = skimage.morphology.label(best_corrected_labels, connectivity=1)
    best_corrected_labels = best_corrected_labels.astype(np.uint16)

    return best_corrected_labels


# For each vesicle, make a surrounded sphere cut out a box around the region in the labelled
# I defined radius as half of largest aspect to surround whole vesicle
# At the end put the sphere instead of vesicles
# you can easyly detrmine how dilation of erroies u want aslo

def elipsoid_vesicles(image_label, diOrEr=0):
    '''Optimize the size of each vesicle mask to minimize shell intensity'''

    # calculate the properties of the labelled regions
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('centroid', 'label', 'bbox'))

    corrected_labels = np.zeros(image_label.shape)
    print(np.shape(corrected_labels))
    for i in range(1, len(vesicle_regions['label'])):

        # here we can have improvment instead of using center fit them into box!
        all_r = []
        p = int(round((vesicle_regions['bbox-3'][i] - vesicle_regions['bbox-0'][i]) / 2))
        q = int(round((vesicle_regions['bbox-4'][i] - vesicle_regions['bbox-1'][i]) / 2))
        r = int(round((vesicle_regions['bbox-5'][i] - vesicle_regions['bbox-2'][i]) / 2))

        radius = np.max([p, q, r]) + diOrEr
        all_r += radius
        #         print(p,q,r)


        #         print(np.shape(sub_int_im))

        sub_label = image_label[vesicle_regions['centroid-0'][i] - radius:vesicle_regions['centroid-0'][i] + radius + 1,
                    vesicle_regions['centroid-1'][i] - radius:vesicle_regions['centroid-1'][i] + radius + 1,
                    vesicle_regions['centroid-2'][i] - radius:vesicle_regions['centroid-2'][i] + radius + 1]
        sub_label = sub_label == vesicle_regions['label'][i]

        image_label_opt = skimage.morphology.ball(radius)
        #         fig = plt.figure()
        #         ax = fig.gca(projection='3d')
        #         voxels=image_label_opt
        #         ax.voxels(voxels, edgecolor='k')
        #         break

        # complete the full image with that sub-image
        if (vesicle_regions['centroid-0'][i] + radius < np.shape(image_label)[0]) and (
                vesicle_regions['centroid-1'][i] + radius < np.shape(image_label)[1]) and (
                vesicle_regions['centroid-2'][i] + radius < np.shape(image_label)[2]):
            sub_label = image_label_opt
            best_mask = sub_label
            px, py, pz = np.where(best_mask)
            corrected_labels[px + vesicle_regions['centroid-0'][i] - radius,
                             py + vesicle_regions['centroid-1'][i] - radius,
                             pz + vesicle_regions['centroid-2'][i] - radius] = i
        else:
            best_mask = sub_label
            px, py, pz = np.where(best_mask)
            corrected_labels[px + vesicle_regions['centroid-0'][i] - radius,
                             py + vesicle_regions['centroid-1'][i] - radius,
                             pz + vesicle_regions['centroid-2'][i] - radius] = i

    corrected_labels = skimage.morphology.label(corrected_labels >= 1, connectivity=1)
    corrected_labels = corrected_labels.astype(np.uint16)

    corrected_labels = oneToOneCorrection(image_label, corrected_labels)
    # print(all_r)
    return corrected_labels








