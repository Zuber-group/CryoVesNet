import sys
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.registration
import numpy as np
import pandas as pd
import os
import mrcfile
from scipy import ndimage
from scipy.spatial.distance import mahalanobis
from scipy import interpolate
from tqdm import tqdm
from . import pipeline
from pathlib import Path
import math
from . import evaluation_class
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

def get_voxel_size_in_nm(path_to_file):
    with mrcfile.open(path_to_file, header_only=True) as tomo:
        voxel_size = float(tomo.voxel_size.x)/10
    return voxel_size

def min_radius_of_vesicle(voxel_size, radius_thr=12):
    ''' Here we assume that the minimum radius of vesicle is 12 nm '''
    radius = radius_thr / voxel_size
    return radius

def min_volume_of_vesicle(voxel_size, radius_thr = 12):
    "Calculate the minimum volume of vesicle in voxels"
    radius = min_radius_of_vesicle(voxel_size, radius_thr)
    volume_of_vesicle = (4.0 / 3.0) * np.pi * (radius) ** 3
    return volume_of_vesicle

def save_label_to_mrc(labels,path_to_file,template_path=None,q = 1):
    """
    using pathlib.Path, there is no need to do these complicated string operations
    hence BZ changed input parameters to only labels and output path.
    :param labels:
    :param path_to_file:
    :param template_path: if given, sets output voxel size and origin as in template file
    """
    with mrcfile.new(path_to_file, overwrite=True) as mrc:
        mrc.set_data(labels)
        if template_path:
            template = mrcfile.open(template_path, header_only=True)
            #for a reason that is not understood, origin needs to be inverted
            mrc.header['origin'].x = template.header['origin'].x * q
            mrc.header['origin'].y = template.header['origin'].y * q
            mrc.header['origin'].z = template.header['origin'].z * q
            mrc.voxel_size = template.voxel_size
    return path_to_file

def decompress_mrc(input, output=None):
    input = Path(input)
    if not output:
        output = input.with_suffix('')
    with mrcfile.new(output, overwrite=True) as output_mrc:
        with mrcfile.open(input) as input_mrc:
            output_mrc.header['origin'] = input_mrc.header['origin']
            output_mrc.voxel_size = input_mrc.voxel_size
            output_mrc.set_data(input_mrc.data)

def save_label_to_tiff(labels,path_to_file,folder_to_save,suffix):
    skimage.io.imsave(os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.tiff',
                      labels, plugin='tifffile')
    return os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + suffix+'.tiff'

# here we use mask_image (probabilty map) comes from our network
# for each vesicles we compute theroshold from this map
# if you set dilation number you will have more dilated vesicles
# if you set convex as 1 then all the pacman gona die :D

def find_threshold_per_vesicle_intensity(int_image, image_label, mask_image, minimum_volume_of_vesicle):
    '''Optimize the size of each vesicle mask to minimize shell intensity'''

    # calculate the properties of the labelled regions
    vesicle_regions = skimage.measure.regionprops_table(image_label,
                                                        properties=('label', 'bbox'))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    mean_shell_arg = []
    # print(int_image.shape,type(int_image.shape),np.shape(int_image))
    box_extension = 20
    corrected_labels = np.zeros(int_image.shape, dtype=np.int16)
    for i in tqdm(range(len(labels)), desc='finding optimal threshold per vesicle'):
        extended_bbox = get_extended_bbox(bboxes[i], box_extension)
        clipped_bbox = clip_box_to_image_size(int_image, extended_bbox)
        clipped_bbox_center = get_center_of_bbox(clipped_bbox)
        sub_int_im = extract_box(int_image, clipped_bbox)
        sub_mask = extract_box(mask_image, clipped_bbox)
        old_label = extract_box(image_label, clipped_bbox)
        old_label_mask = old_label == labels[i]
        new_opt_thr, new_mean_shell_val = my_threshold(sub_int_im, sub_mask)
        omit = 0
        if (np.array(sub_mask.shape) != 0).all():
            image_label_opt = skimage.morphology.label(sub_mask >= new_opt_thr)
            sub_label = image_label_opt
            if (image_label_opt[tuple(clipped_bbox_center)] == 0):
                center_of_mass = ndimage.measurements.center_of_mass(image_label_opt, vesicle_regions['label'][i])
                if not any(np.isnan(center_of_mass)):
                    center_of_mass = np.array(center_of_mass, dtype=int)
                    sub_label = image_label_opt
                    new_label_of_ves = image_label_opt[tuple(center_of_mass)]
                    if new_label_of_ves != 0:
                        sub_label_mask = sub_label == new_label_of_ves
                        best_mask = sub_label_mask
                    else:
                        print('out')
                        best_mask = old_label_mask.copy()
                else:
                    print(f"bad vesicle {labels[i]}")
                    best_mask = old_label_mask.copy()
                    omit = 1
            else:
                #good vesicle
                sub_label_mask = sub_label == image_label_opt[tuple(clipped_bbox_center)]
                best_mask = sub_label_mask
        else:
            print(f"bad vesicle {labels[i]}")
            best_mask = old_label_mask.copy()
            omit = 1
        if omit == 0 and len(best_mask[best_mask > 0]) < minimum_volume_of_vesicle:
            print('small')
            omit = 1

        best_param, best_mask = adjust_shell_intensity(sub_int_im, best_mask)

        mean_shell_arg.append(new_mean_shell_val)

        if omit == 0:
            px, py, pz = np.where(best_mask)
            corrected_labels[px + clipped_bbox[0,0],
                             py + clipped_bbox[1,0],
                             pz + clipped_bbox[2,0]] = labels[i]

    corrected_labels = skimage.morphology.label(corrected_labels >= 1, connectivity=1)
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels, mean_shell_arg


def pacman_killer(image_label):
    "This function is used to make vesicles convex"
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('label', 'bbox'))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    #BZ: I don't think there is any reason to extend the box for this function
    #I left the option to do it (just change extension to whatever desired)
    box_extension = 0
    corrected_labels = np.zeros(image_label.shape)
    for i in tqdm(range(len(labels)), desc='making vesicles convex'):
        extended_bbox = get_extended_bbox(bboxes[i], box_extension)
        clipped_bbox = clip_box_to_image_size(image_label, extended_bbox)
        old_label = extract_box(image_label, clipped_bbox)
        old_label_mask = old_label == labels[i]
        try:
            old_label_mask = skimage.morphology.convex_hull_image(old_label_mask, offset_coordinates=False)
        except:
            print("?", vesicle_regions['label'][i])
        px, py, pz = np.where(old_label_mask)
        corrected_labels[px + clipped_bbox[0,0],
                         py + clipped_bbox[1,0],
                         pz + clipped_bbox[2,0]] = labels[i]
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels

def fast_pacman_killer(image_label):
    "In some cases the convex_hull_image function fails, which in pacman killer we used try and catch but normally this whould be faster"
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('label', 'bbox'))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    #BZ: I don't think there is any reason to extend the box for this function
    #I left the option to do it (just change extension to whatever desired)
    box_extension = 0
    corrected_labels = np.zeros(image_label.shape)
    for i in tqdm(range(len(labels)), desc='making vesicles convex'):
        extended_bbox = get_extended_bbox(bboxes[i], box_extension)
        clipped_bbox = clip_box_to_image_size(image_label, extended_bbox)
        old_label = extract_box(image_label, clipped_bbox)
        old_label_mask = old_label == labels[i]
        old_label_mask = skimage.morphology.convex_hull_image(old_label_mask, offset_coordinates=False)
        px, py, pz = np.where(old_label_mask)
        corrected_labels[px + clipped_bbox[0,0],
                         py + clipped_bbox[1,0],
                         pz + clipped_bbox[2,0]] = labels[i]
    corrected_labels = corrected_labels.astype(np.uint16)
    return corrected_labels

def my_threshold(image, image_mask,min_thr=0.8,max_thr=1,step=0.01):
    # first, calculate shell-intensity at different thresholds
    shell_pixels = []
    # for th in tqdm(np.arange(0.8, 1, 0.1), desc='finding global threshold on unet mask'):
    for th in np.arange(min_thr, max_thr, step):
        image_mask_bin = np.zeros(image_mask.shape)
        image_mask_bin[image_mask > th] = 1

        image_mask_bin_erode = skimage.morphology.binary_erosion(image_mask_bin, footprint=skimage.morphology.ball(1))
        image_shell = image_mask_bin - image_mask_bin_erode

        shell_pixels.append(image[image_shell.astype(bool)])

    mean_shell_val = [np.mean(x) for x in shell_pixels]
    # find optimal threshold
    opt_th = np.arange(min_thr, max_thr, step)[np.argmin(mean_shell_val)]
    return opt_th, mean_shell_val

def adjust_shell_intensity(image_int, image_labels):
    '''For a given sub-region adjust the size the vesicle mask to minimize its average shell intensity'''
    mean_shell = []
    # calculate the mean shell value in the original segmentation.
    mask_adjusted = image_labels.copy()
    # create a slightly eroded version of the region
    image_mask_bin_erode = skimage.morphology.binary_erosion(mask_adjusted, footprint=skimage.morphology.ball(1))
    # combine original image and eroded one to keep only the shell
    image_shell = mask_adjusted ^ image_mask_bin_erode
    # recover pixels in the shell and calculate their mean intensity
    shell_pixels = image_int[image_shell.astype(bool)]
    mean_shell.append([0, np.mean(shell_pixels)])
    best_param = 0
    best_value = np.mean(shell_pixels)
    best_mask = image_labels.copy()
    for i in range(1, 8):
        mask_adjusted = skimage.morphology.binary_dilation(image_labels, footprint=skimage.morphology.ball(i))
        image_mask_bin_erode = skimage.morphology.binary_erosion(mask_adjusted, footprint=skimage.morphology.ball(1))
        image_shell = mask_adjusted ^ image_mask_bin_erode
        shell_pixels = image_int[image_shell.astype(bool)]
        meanval = np.mean(shell_pixels)
        mean_shell.append([i, meanval])
        if meanval < best_value:
            best_value = meanval
            best_param = i
            best_mask = mask_adjusted
    return best_param, best_mask  # , mean_shell

def objectwise_evalution(reff, prediction, delta_size=1, proportion=0.0):
    # You do not need this function , this is just for evaluation
    reff = reff.copy()
    prediction = prediction.copy()
    reff_regions = skimage.measure.regionprops_table(reff, properties=('centroid', 'label', 'bbox'))
    predict_regions = skimage.measure.regionprops_table(prediction, properties=('centroid', 'label', 'bbox'))

    all = []
    TP = 0
    FN = 0
    print(np.shape(reff))
    for i in range(0, len(reff_regions['label'])):
        sub_old_label = reff[
                        reff_regions['bbox-0'][i] - delta_size: reff_regions['bbox-3'][i] + delta_size + 1,
                        reff_regions['bbox-1'][i] - delta_size: reff_regions['bbox-4'][i] + delta_size + 1,
                        reff_regions['bbox-2'][i] - delta_size: reff_regions['bbox-5'][i] + delta_size + 1]
        sub_new_label = prediction[
                        reff_regions['bbox-0'][i] - delta_size: reff_regions['bbox-3'][i] + delta_size + 1,
                        reff_regions['bbox-1'][i] - delta_size: reff_regions['bbox-4'][i] + delta_size + 1,
                        reff_regions['bbox-2'][i] - delta_size: reff_regions['bbox-5'][i] + delta_size + 1]

        sub_old_label_mask = sub_old_label == reff_regions['label'][i]
        p = int(round((reff_regions['bbox-3'][i] - reff_regions['bbox-0'][i])))
        q = int(round((reff_regions['bbox-4'][i] - reff_regions['bbox-1'][i])))
        r = int(round((reff_regions['bbox-5'][i] - reff_regions['bbox-2'][i])))

        reff_diameter = np.max([p, q, r])
        px, py, pz = np.where(sub_old_label_mask)
        areaOfPrediction = sub_new_label[px, py, pz]
        reff_center = np.array(
            [reff_regions['centroid-0'][i], reff_regions['centroid-1'][i], reff_regions['centroid-2'][i]])
        # print(type(areaOfPrediction))

        unique, counts = np.unique(areaOfPrediction, return_counts=True)
        # print(unique)
        # print(counts)
        areaOfPrediction = np.delete(areaOfPrediction, np.where(areaOfPrediction == 0))
        # (filter(lambda a: a != 2, areaOfPrediction))
        unique, counts = np.unique(areaOfPrediction, return_counts=True)
        # print(unique,counts)
        if len(unique) > 0:
            if counts[-1] >= proportion * np.count_nonzero(sub_old_label_mask):
                TP = TP + 1
                related_label = unique[-1]

                index_related_label = np.where(predict_regions['label'] == related_label)[0][0]
                # print(related_label,index_related_label)
                p = int(round(
                    (predict_regions['bbox-3'][index_related_label] - predict_regions['bbox-0'][index_related_label])))
                q = int(round(
                    (predict_regions['bbox-4'][index_related_label] - predict_regions['bbox-1'][index_related_label])))
                r = int(round(
                    (predict_regions['bbox-5'][index_related_label] - predict_regions['bbox-2'][index_related_label])))

                predicted_diameter = np.max([p, q, r])
                predicted_center = np.array([predict_regions['centroid-0'][index_related_label],
                                             predict_regions['centroid-1'][index_related_label],
                                             predict_regions['centroid-2'][index_related_label]])
                a = min(reff_diameter, predicted_diameter)
                b = max(reff_diameter, predicted_diameter)
                c = 1 - a / b
                all += [[reff_regions['label'][i], related_label, reff_diameter, predicted_diameter, c,
                         math.sqrt(sum((reff_center - predicted_center) ** 2))]]
                # all+=[abs(reff_center - predicted_center)]
                # print("Here")
                # print(reff_center)
                # print(predicted_center)
                # print((np.array(reff_center) - np.array(predicted_center)))
            else:
                # print(str(reff_regions['label'][i]) + "!!!!!!!!!!!!!!!!!!!!!!!!!!")
                predicted_diameter = -1
                predicted_center = reff_center
                FN = FN + 1
                qx, qy, qz = np.where(reff == reff_regions['label'][i])
                reff[qx, qy, qz] = 0
        else:
            # print(str(reff_regions['label'][i]) + "!!!!!!!!!!!!!!!!!!!!!!!!!!")
            predicted_diameter = -1
            predicted_center = reff_center
            FN = FN + 1
            qx, qy, qz = np.where(reff == reff_regions['label'][i])
            reff[qx, qy, qz] = 0
        # print(reff_diameter, predicted_diameter)
        # unique = np.unique(prediction[px, py, pz])

    evaluator = evaluation_class.ConfusionMatrix(reff > 0, prediction > 0)
    print(evaluator.former_dice())
    print(len(reff_regions['label']))
    print(TP)
    print(FN)
    tab = np.array(all)
    # print(np.shape(tab))
    # print(tab[:, 0])
    # print(tab[:, 1])
    # print(tab[:, 2])
    # print(np.shape(tab[:,0]))
    # print(ttest_1samp(tab[:,0],0))
    # # print(tab)
    # print(np.mean(tab[:,4]))
    important=[evaluator.former_dice(), len(reff_regions['label']) ,TP, FN, np.mean(tab[:, 4]),np.mean(tab[:, 5]),np.std(tab[:, 5])]
    # important = [np.mean(tab, axis=0).tolist(), np.std(tab, axis=0).tolist(), ttest_1samp(tab[:, 0], 0),
    #            ttest_1samp(tab[:, 1], 0), ttest_1samp(tab[:, 2], 0), ttest_ind(tab[:, 0], tab[:, 1]),
    #             ttest_ind(tab[:, 0], tab[:, 2]), ttest_ind(tab[:, 1], tab[:, 2])]
    # important = [np.mean(tab, axis=0).tolist(), np.std(tab, axis=0).tolist()]
    return important


def new_objectwise_evalution(reff, prediction, delta_size=1, proportion=0.0):
    # You do not need this function , this is just for evaluation
    reff = reff.copy()
    prediction = prediction.copy()
    reff_regions = skimage.measure.regionprops_table(reff, properties=('centroid', 'label', 'bbox'))
    predict_regions = skimage.measure.regionprops_table(prediction, properties=('centroid', 'label', 'bbox'))
    good = np.zeros_like(prediction)

    all = []
    TP = 0
    FN = 0
    print(np.shape(reff))
    for i in range(0, len(reff_regions['label'])):
        sub_old_label = reff[
                        reff_regions['bbox-0'][i] - delta_size: reff_regions['bbox-3'][i] + delta_size + 1,
                        reff_regions['bbox-1'][i] - delta_size: reff_regions['bbox-4'][i] + delta_size + 1,
                        reff_regions['bbox-2'][i] - delta_size: reff_regions['bbox-5'][i] + delta_size + 1]
        sub_new_label = prediction[
                        reff_regions['bbox-0'][i] - delta_size: reff_regions['bbox-3'][i] + delta_size + 1,
                        reff_regions['bbox-1'][i] - delta_size: reff_regions['bbox-4'][i] + delta_size + 1,
                        reff_regions['bbox-2'][i] - delta_size: reff_regions['bbox-5'][i] + delta_size + 1]

        sub_old_label_mask = sub_old_label == reff_regions['label'][i]
        p = int(round((reff_regions['bbox-3'][i] - reff_regions['bbox-0'][i])))
        q = int(round((reff_regions['bbox-4'][i] - reff_regions['bbox-1'][i])))
        r = int(round((reff_regions['bbox-5'][i] - reff_regions['bbox-2'][i])))

        reff_diameter = np.max([p, q, r])
        px, py, pz = np.where(sub_old_label_mask)
        areaOfPrediction = sub_new_label[px, py, pz]
        reff_center = np.array(
            [reff_regions['centroid-0'][i], reff_regions['centroid-1'][i], reff_regions['centroid-2'][i]])
        # print(type(areaOfPrediction))

        unique, counts = np.unique(areaOfPrediction, return_counts=True)
        # print(unique)
        # print(counts)
        areaOfPrediction = np.delete(areaOfPrediction, np.where(areaOfPrediction == 0))
        # (filter(lambda a: a != 2, areaOfPrediction))
        unique, counts = np.unique(areaOfPrediction, return_counts=True)
        # print(unique,counts)
        is_tp=False
        if len(unique) > 0:
            if counts[-1] >= proportion * np.count_nonzero(sub_old_label_mask):

                related_label = unique[-1]

                index_related_label = np.where(predict_regions['label'] == related_label)[0][0]
                # print(related_label,index_related_label)
                p = int(round(
                    (predict_regions['bbox-3'][index_related_label] - predict_regions['bbox-0'][index_related_label])))
                q = int(round(
                    (predict_regions['bbox-4'][index_related_label] - predict_regions['bbox-1'][index_related_label])))
                r = int(round(
                    (predict_regions['bbox-5'][index_related_label] - predict_regions['bbox-2'][index_related_label])))

                predicted_diameter = np.max([p, q, r])
                predicted_center = np.array([predict_regions['centroid-0'][index_related_label],
                                             predict_regions['centroid-1'][index_related_label],
                                             predict_regions['centroid-2'][index_related_label]])

                d = math.sqrt(sum((reff_center - predicted_center) ** 2))

                if (d<= min(predicted_diameter,reff_diameter)/2):
                    TP = TP + 1
                    is_tp = True
                    a = min(reff_diameter, predicted_diameter)
                    b = max(reff_diameter, predicted_diameter)
                    c = 1 - a / b
                    all += [[reff_regions['label'][i], related_label, reff_diameter, predicted_diameter, c,
                             d]]
        if not is_tp:
            FN = FN + 1
            qx, qy, qz = np.where(reff == reff_regions['label'][i])
            reff[qx, qy, qz] = 0

    FP = len(predict_regions['label']) -TP

    evaluator = evaluation_class.ConfusionMatrix(reff > 0, prediction > 0)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*recall*precision/(recall+precision)
    tab = np.array(all)
    important = [evaluator.former_dice(), len(reff_regions['label']) ,len(predict_regions['label']), TP, FN,FP, recall, precision, f1,
                 np.mean(tab[:, 4]), np.mean(tab[:, 5]),np.std(tab[:, 5])]
    # important = [np.mean(tab, axis=0).tolist(), np.std(tab, axis=0).tolist(), ttest_1samp(tab[:, 0], 0),
    #            ttest_1samp(tab[:, 1], 0), ttest_1samp(tab[:, 2], 0), ttest_ind(tab[:, 0], tab[:, 1]),
    #             ttest_ind(tab[:, 0], tab[:, 2]), ttest_ind(tab[:, 1], tab[:, 2])]
    # important = [np.mean(tab, axis=0).tolist(), np.std(tab, axis=0).tolist()]
    return important


def oneToOneCorrection(old_label, new_label, delta_size=3):
    """
    this function give 2 label map OLD and NEW
    and correct anything you want not happend in new one which is not desired and turn it back to old one
    here we had intersection problem of vesicles after changing threshold
    at the end this function OR 2 given input (which is source of some bug =
    we should bring it to top -> later decision needed in group)
    but give u larger map as result
    :param old_label:
    :param new_label:
    :return:
    """
    vesicle_regions = skimage.measure.regionprops_table(old_label, properties=('centroid', 'label', 'bbox'))
    myvesicle_regions = skimage.measure.regionprops_table(new_label, properties=('centroid', 'label', 'bbox'))
    mean_shell_arg = []
    #     corrected_labels = np.zeros(new_label.shape)
    corrected_labels = new_label.copy()
    vesicles = []
    for i in range(1, len(vesicle_regions['label'])):
        sub_old_label = old_label[vesicle_regions['bbox-0'][i] - delta_size:vesicle_regions['bbox-3'][i] + delta_size+1,
                        vesicle_regions['bbox-1'][i] - delta_size:vesicle_regions['bbox-4'][i] + delta_size + 1,
                        vesicle_regions['bbox-2'][i] - delta_size:vesicle_regions['bbox-5'][i] + delta_size + 1]
        sub_old_label_mask = sub_old_label == vesicle_regions['label'][i]

        sub_new_label = new_label[vesicle_regions['bbox-0'][i] - delta_size:vesicle_regions['bbox-3'][i] + delta_size + 1,
                        vesicle_regions['bbox-1'][i] - delta_size:vesicle_regions['bbox-4'][i] + delta_size + 1,
                        vesicle_regions['bbox-2'][i] - delta_size:vesicle_regions['bbox-5'][i] + delta_size + 1]
        a, b, c = ndimage.measurements.center_of_mass(sub_old_label_mask, vesicle_regions['label'][i])
        if any(np.isnan([a, b, c])):
            vesicles += [0]
        else:
            a = int(a)
            b = int(b)
            c = int(c)
            vesicles += [sub_new_label[a, b, c]]
    unique, counts = np.unique(vesicles, return_counts=True)
    intersected = (unique[counts > 1])
    for j in intersected[1:]:
        places = np.where(vesicles == j)
        places = places[0]
        sub_temp_label = new_label[myvesicle_regions['bbox-0'][j - 1] - delta_size:myvesicle_regions['bbox-3'][j - 1] + delta_size + 1,
                         myvesicle_regions['bbox-1'][j - 1] - delta_size:myvesicle_regions['bbox-4'][j - 1] + delta_size + 1,
                         myvesicle_regions['bbox-2'][j - 1] - delta_size:myvesicle_regions['bbox-5'][j - 1] + delta_size + 1]
        sub_temp_label = sub_temp_label == j
        px, py, pz = np.where(sub_temp_label)
        corrected_labels[px + myvesicle_regions['bbox-0'][j - 1] - delta_size,
                         py + myvesicle_regions['bbox-1'][j - 1] - delta_size,
                         pz + myvesicle_regions['bbox-2'][j - 1] - delta_size] = 0
        for m in places:
            #             print(m)
            sub_orig_label = old_label[vesicle_regions['bbox-0'][m] - delta_size:vesicle_regions['bbox-3'][m] + delta_size + 1,
                             vesicle_regions['bbox-1'][m] - delta_size:vesicle_regions['bbox-4'][m] + delta_size + 1,
                             vesicle_regions['bbox-2'][m] - delta_size:vesicle_regions['bbox-5'][m] + delta_size + 1]
            sub_orig_label = sub_orig_label == vesicle_regions['label'][m]
            px, py, pz = np.where(sub_orig_label)
            corrected_labels[px + vesicle_regions['bbox-0'][m] - delta_size,
                             py + vesicle_regions['bbox-1'][m] - delta_size,
                             pz + vesicle_regions['bbox-2'][m] - delta_size] = 1000 + m
    best_corrected_labels = (corrected_labels >= 1) | (old_label >= 1)
    best_corrected_labels = skimage.morphology.label(best_corrected_labels, connectivity=1)
    best_corrected_labels = best_corrected_labels.astype(np.uint16)
    return best_corrected_labels

def get_sphere_dataframe(image, image_label, margin=5,tight=False,keep_ellipsoid=False):
    corrected_labels = np.zeros(image_label.shape, dtype=int)
    image_bounding_box = get_image_bounding_box(image_label)
    vesicle_regions = pd.DataFrame(skimage.measure.regionprops_table(image_label,
                                                                     properties=('centroid', 'label', 'bbox','axis_major_length','axis_minor_length')))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    centroids = get_centroids_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    thicknesses, densities, radii, centers, kept_labels,my_radial = [],[],[],[],[],[]
    ellipsoid_tags= []
    for i in tqdm(range(len(vesicle_regions)), desc="fitting sphere to vesicles"):
        label = labels[i]
        radius = get_label_largest_radius(bboxes[i])  # this is an integer
        rounded_centroid = np.round(centroids[i]).astype(np.int16)  # this is an array of integers
        # eccentricity= vesicle_regions['axis_major_length'][i]/vesicle_regions['axis_minor_length'][i]
        eccentricity = math.sqrt(
            1 - (vesicle_regions['axis_minor_length'][i] / vesicle_regions['axis_major_length'][i]) ** 2)
        if not keep_ellipsoid or eccentricity <= 0.48:
            density, keep_label, new_centroid, new_radius, thickness, radial = get_sphere_parameters(image, label, margin, radius,
                                                                                             rounded_centroid,tight=tight)

            if keep_label:
                thicknesses.append(thickness)
                densities.append(density)
                radii.append(new_radius)
                centers.append(new_centroid)
                kept_labels.append(label)
                my_radial.append(radial)
    if keep_ellipsoid:
        for i in tqdm(range(len(vesicle_regions)), desc="Identify the ellipsoid vesicles"):
            label = labels[i]
            radius = get_label_largest_radius(bboxes[i])  # this is an integer
            rounded_centroid = np.round(centroids[i]).astype(np.int16)  # this is an array of integers
            eccentricity = math.sqrt(1 - (vesicle_regions['axis_minor_length'][i] / vesicle_regions['axis_major_length'][i]) ** 2)
            if eccentricity > 0.48 and eccentricity < 0.95:
                density, keep_label, new_centroid, new_radius, thickness, radial = get_sphere_parameters(image, label,
                                                                                                     2, radius,
                                                                                                     rounded_centroid,max_cycles=1,
                                                                                                     tight=tight)
                if keep_label:
                    # check if density and thickness and radius are not outlier based on zscore of thicknesses densities radii:
                    mean_thickness = np.mean(thicknesses)
                    std_thickness = np.std(thicknesses)

                    mean_density = np.mean(densities)
                    std_density = np.std(densities)

                    mean_radius = np.mean(radii)
                    std_radius = np.std(radii)

                    threshold = 3

                    if abs(density - mean_density) / std_density < threshold and abs(
                            radius - mean_radius) / std_radius < threshold+1:
                                ellipsoid_tags.append(label)




    df = pd.DataFrame(zip(kept_labels, thicknesses, densities, radii, centers),
                          columns=['label','thickness','density','radius','center'])
    df = df.set_index('label')
    return df,my_radial,ellipsoid_tags


def get_sphere_parameters(image, label, margin, radius, rounded_centroid,max_cycles = 10, tight=False):
    try:
        shift, new_radius = get_optimal_sphere_position_and_radius(image, rounded_centroid, radius, margin=margin, max_cycles=max_cycles, tight=tight)
        new_centroid = (rounded_centroid - shift).astype(np.int16)
        image_box = extract_box_of_radius(image, new_centroid, radius + margin)
        thickness, density, radial = get_sphere_membrane_thickness_and_density_from_image(image_box)
        keep_label = True

        # if thickness < 6:
        #     print(f"small thickness, label {label}")
    except ValueError:
    # else:

        keep_label = False
        radial= np.zeros(100)
        thickness = np.nan
        density = np.nan
        new_radius = np.nan
        new_centroid = rounded_centroid
    return density, keep_label, new_centroid, new_radius, thickness,radial


def mahalanobis_distances(df, axis=0):
    '''
    Returns a pandas Series with Mahalanobis distances for each sample on the
    axis.

    Note: does not work well when # of observations < # of dimensions
    Will either return NaN in answer
    or (in the extreme case) fail with a Singular Matrix LinAlgError

    Args:
        df: pandas DataFrame with columns to run diagnostics on
        axis: 0 to find outlier rows, 1 to find outlier columns
    copyright @ http://github.com/tyarkoni/pliers
    '''
    df = df.transpose() if axis == 1 else df
    means = df.mean()
    try:
        inv_cov = np.linalg.inv(df.cov())
    except LinAlgError:
        return pd.Series([np.NAN] * len(df.index), df.index,
                         name='Mahalanobis')
    dists = []
    for i, sample in df.iterrows():
        dists.append(mahalanobis(sample, means, inv_cov))
    return pd.Series(dists, df.index, name='Mahalanobis')

def make_vesicle_from_sphere_dataframe(image_label, sphere_df):
    corrected_labels = np.zeros(np.array(image_label).shape).astype(np.int16)
    sphere_dict = {}
    for label, row in sphere_df.iterrows():
        add_sphere_in_dict(sphere_dict, row.radius)
        put_spherical_label_in_array(corrected_labels, row.center, row.radius, label,inplace=True)
    return corrected_labels

def add_sphere_in_dict(sphere_dict, radius):
    if radius in sphere_dict:
        return
    sphere_dict[radius] = skimage.morphology.ball(radius)

def remove_labels_under_points(image_label, points_to_remove):
    labels_to_remove = np.unique([image_label[a,b,c] for a,b,c in np.round(points_to_remove).astype(np.int16)])
    selection_mask = np.isin(image_label, labels_to_remove)
    corrected_labels = image_label.copy()
    corrected_labels[selection_mask] = 0 #for figure
    return corrected_labels



def expand_small_labels(deep_mask, labels, initial_threshold, min_vol,p,q,t):
    '''Expand labels until they are q times bigger than min_vol'''
    expanded_labels = labels.copy()
    vesicle_regions = pd.DataFrame(skimage.measure.regionprops_table(labels, properties=('label', 'area', 'centroid')))
    small_labels = vesicle_regions[vesicle_regions.area < p*min_vol].set_index('label')
    step = 0.025
    for threshold in tqdm(np.arange((initial_threshold-step), 0.8 , -step), desc="Expanding labels until none is too small"):
        labels = skimage.morphology.label(deep_mask>threshold)
        new_vesicle_regions = pd.DataFrame(skimage.measure.regionprops_table(labels, properties=('label', 'area')))
        new_vesicle_regions.set_index('label', inplace=True)
        small_labels_fixed = []
        for label, row in small_labels.iterrows():
            centroid = (row['centroid-0'],row['centroid-1'],row['centroid-2'])
            centroid = tuple(np.array(centroid).astype(np.int16))
            new_label = labels[centroid]
            if (new_label) == 0:
                pass
            else:
                new_vol = new_vesicle_regions.loc[new_label].area
                if new_vol >= q * min_vol:
                    expanded_labels[np.where(labels==new_label)] = label
                    small_labels_fixed.append(label)
        small_labels = small_labels.drop(labels=small_labels_fixed)
        if len(small_labels) == 0 :
            break
    return expanded_labels, small_labels


def vesicles_table(labels):
    ves_tabel = pd.DataFrame(
        skimage.measure.regionprops_table(labels, properties=('label', 'area', 'centroid', 'bbox', 'extent')))
    ves_tabel['area_zscore'] = (ves_tabel['area'] - ves_tabel['area'].mean()) / ves_tabel['area'].std(ddof=0)
    ves_tabel['extent_zscore'] = (ves_tabel['extent'] - ves_tabel['extent'].mean()) / ves_tabel['extent'].std(ddof=0)

    print("Tabel computed!")
    return ves_tabel

def collision_solver_debug(deep_mask, deep_labels,ves_table, threshold ,delta_size = 1):
    '''We assume that the have low extent value and high area_zscore are colliding vesicles
    Later on we goes back to mask and search for finer threshold to separate them'''

    collision_ves = ves_table[(ves_table['extent'] < 0.5) & (ves_table['area_zscore'] > 1)]
    print(collision_ves)
    old_mask = deep_mask.copy()
    old_label = deep_labels.copy()
    for i in collision_ves.iterrows():
        # print(i)
        sub_old_mask = old_mask[
                       collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
                       collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
                       collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1]

        sub_old_label = old_label[
                        collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
                        collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
                        collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1]

        sub_old_label_mask = sub_old_label != collision_ves['label'][i[0]]
        pxj, pyj, pzj = np.where(sub_old_label_mask)
        sub_old_mask[pxj, pyj, pzj] = 0
        thr = threshold
        pre_labels, pre_nc = skimage.morphology.label(sub_old_mask > thr, return_num=True, connectivity=None)
        # print(pre_nc)
        is_break = 0
        base_thr = thr
        step = 0.01

        while not is_break:
            for th in np.arange(base_thr, 1, step):
                # temp=sub_old_label_mask>th
                temp = sub_old_mask > th
                labels, nc = skimage.morphology.label(temp, return_num=True, connectivity=None)
                if nc > pre_nc:
                    is_break = 1
                    # print(pre_nc, nc)
                    if collision_ves['label'][i[0]] == 274:
                        print(np.shape(sub_old_mask), np.shape(sub_old_label),np.shape(temp))
                        print(i,th)
                    px, py, pz = np.where(temp > 0)

                    pxq, pyq, pzq = np.where(~sub_old_label_mask)

                    old_label[pxq + collision_ves['bbox-0'][i[0]] - delta_size,
                              pyq + collision_ves['bbox-1'][i[0]] - delta_size,
                              pzq + collision_ves['bbox-2'][i[0]] - delta_size] = 0
                    # old_label[collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
                    # collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
                    # collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1] = labels

                    old_label[px + collision_ves['bbox-0'][i[0]] - delta_size,
                              py + collision_ves['bbox-1'][i[0]] - delta_size,
                              pz + collision_ves['bbox-2'][i[0]] - delta_size] = collision_ves['label'][i[0]] + 1000
                    break
            base_thr = 1-step
            step = step/10
            if step == 0.0001:
                break



    #
    #
    # pl2.quick_setup(['deep_mask','deep_labels'])
    old_label = skimage.morphology.label(old_label, connectivity=1)
    # old_label = skimage.morphology.label(old_label, connectivity=1)
    old_label = old_label.astype(np.uint16)

    return old_label


def collision_solver(deep_mask, deep_labels, ves_table, threshold, delta_size=1):
    EXTENT_THRESHOLD = 0.5
    AREA_ZSCORE_THRESHOLD = 1
    LABEL_OFFSET = 1000
    DEBUG_LABEL = 274

    collision_ves = ves_table[
        (ves_table['extent'] < EXTENT_THRESHOLD) & (ves_table['area_zscore'] > AREA_ZSCORE_THRESHOLD)]
    # Vesicles with low extent and high area_zscore are considered to be colliding
    print("Collision vesicles:")
    print(collision_ves)

    old_mask = deep_mask.copy()
    old_label = deep_labels.copy()

    for idx, ves in collision_ves.iterrows():
        bbox = ves[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5']].values
        sub_slice = tuple(slice(int(bbox[i] - delta_size), int(bbox[i + 3] + delta_size + 1)) for i in range(3))

        sub_old_mask = old_mask[sub_slice[0], sub_slice[1], sub_slice[2]]
        sub_old_label = old_label[sub_slice[0], sub_slice[1], sub_slice[2]]

        # Create a mask for the current vesicle
        vesicle_mask = sub_old_label == ves['label']

        thr = threshold
        temp_mask = sub_old_mask > thr
        temp_mask[~vesicle_mask] = False  # Only consider the area of the current vesicle
        ttt, pre_nc = skimage.morphology.label(temp_mask, return_num=True, connectivity=None)

        base_thr = thr
        step = 0.01

        while step >= 0.0001:
            for th in np.arange(base_thr, 1, step):
                temp = sub_old_mask > th
                temp[~vesicle_mask] = False  # Only consider the area of the current vesicle
                nc = skimage.morphology.label(temp, return_num=True, connectivity=None)[1]
                if nc > pre_nc:

                    old_label_slice = old_label[sub_slice[0], sub_slice[1], sub_slice[2]]

                    # Only clear the area of the current vesicle
                    old_label_slice[vesicle_mask] = 0

                    # Create new labels
                    new_labels = skimage.morphology.label(temp, connectivity=1)

                    # Update only within the current vesicle area
                    update_mask = (new_labels > 0) & vesicle_mask
                    old_label_slice[update_mask] = new_labels[update_mask] + LABEL_OFFSET + ves['label']

                    # Update the main old_label array
                    old_label[sub_slice[0], sub_slice[1], sub_slice[2]] = old_label_slice

                    break
            else:
                base_thr = 1 - step
                step /= 10
                continue
            break

    return old_label.astype(np.uint16)



def add_sphere_labels_under_points(image, image_labels, points_to_add,
                                   points_to_add_sizes, minimum_box_size):
    corrected_labels = image_labels.copy()
    max_label = image_labels.max()
    for i, point in enumerate(points_to_add):
        rounded_centroid = np.round(point).astype(np.int16)
        point_size = points_to_add_sizes[i]
        radius = int(max(point_size, minimum_box_size)//2)
        print(point_size//2, minimum_box_size//2, radius)
        label = i + max_label + 1
        density, keep_label, new_centroid, new_radius, thickness, radial = \
            get_sphere_parameters(image, label, 5, radius, rounded_centroid, tight=True)
        if keep_label:
            put_spherical_label_in_array(corrected_labels, new_centroid, new_radius, label, inplace = True)
        print(keep_label)
    return corrected_labels

def get_labels_from_regions(vesicle_regions):
    labels = np.array(vesicle_regions['label'])
    return labels


def get_centroids_from_regions(vesicle_regions):
    centroids = []
    for i in range(len(vesicle_regions['centroid-0'])):
        this_centroid = np.zeros(3)
        for j in range(3):
            this_centroid[j] = vesicle_regions[f'centroid-{j}'][i]
        centroids.append(this_centroid)
    centroids = np.array(centroids)
    return centroids



def remove_outliers(deep_labels,ves_table, min_vol ):
    new_label = deep_labels.copy()
    verysmall_vesicles = ves_table[(ves_table['extent'] < 0.25 ) | (ves_table['extent'] > 0.75) | (ves_table['area'] < 1* min_vol)]
    # verysmall_vesicles = ves_table[(ves_table['area'] < 1 * min_vol)]
    verysmall_vesicles = verysmall_vesicles.set_index('label')
    print("Vesicles to remove:")
    print(verysmall_vesicles)
    new_label[np.isin(new_label, verysmall_vesicles.index)] = 0
    return new_label,verysmall_vesicles




def surround_remover(deep_labels,mask,t):
    mask = mask.copy()
    deep_labels_temp = deep_labels.copy()
    mask_invert = 1 - (mask)
    extra = deep_labels_temp * mask_invert
    extra_labels , counts= np.unique(extra, return_counts=True)
    extra_labels = (extra_labels[counts > t])
    extra_labels = extra_labels.tolist()
    extra_labels.remove(0)
    print(extra_labels)

    # for i in extra_labels:
    #     if i == 0:
    #         pass
    #     else:
    #
    #         px, py, pz = np.where(deep_labels== i)
    #         deep_labels_temp[px, py, pz] = 0

    return extra_labels



def adjacent_vesicles(sphere_df):
    new_radii = sphere_df["radius"].values
    new_centroids = sphere_df["center"].values
    new_centroids = new_centroids.tolist()
    new_centroids_arr = [x.tolist() for x in new_centroids]
    new_centroids = np.array(new_centroids_arr)
    rm = np.repeat(np.array((new_radii,)), len(new_radii), axis=0)
    radii_matrix = rm + rm.T
    dm = distance_matrix(new_centroids, new_centroids)
    maxvalue = np.finfo(dm.dtype).max
    for i in range(len(dm)):
        dm[i, i] = maxvalue
    collisions = np.array((np.where(dm <= radii_matrix)))
    return collisions.T

def get_bboxes_from_regions(vesicle_regions):
    bboxes = []
    for i in range(len(vesicle_regions['bbox-0'])):
        this_bbox = np.zeros(6)
        for j in range(6):
            this_bbox[j] = vesicle_regions[f'bbox-{j}'][i]
        this_bbox = this_bbox.reshape((2, 3)).transpose()
        bboxes.append(this_bbox)
    bboxes = np.array(bboxes, dtype = int)
    return bboxes


def put_spherical_label_in_array(array, rounded_centroid, radius, label, inplace = False, sphere=None):
    if sphere is None:
        sphere = skimage.morphology.ball(radius)
    if not inplace:
        array = array.copy()
    px, py, pz = np.where(sphere)
    px = px + rounded_centroid[0] - radius
    py = py + rounded_centroid[1] - radius
    pz = pz + rounded_centroid[2] - radius
    px, py, pz = remove_out_of_range_indices(array, px,py,pz)
    array[px,py,pz] = label
    if not inplace:
        return array

def remove_out_of_range_indices(array,px,py,pz):
    s = array.shape
    indices = np.array((px,py,pz))
    for i in range(3):
        to_remove = list(np.argwhere(indices[i]<0))
        to_remove += list(np.argwhere(indices[i]>s[i]-1))
        indices = np.delete(indices,to_remove,axis=1)
    return indices[0], indices[1], indices[2]

def put_original_label_in_array(array, label_box, label, rounded_centroid, radius, inplace = False):
    label_mask = label_box == label
    px, py, pz = np.where(label_mask)
    if not inplace:
        array = array.copy()
    array[px + rounded_centroid[0] - radius,
          py + rounded_centroid[1] - radius,
          pz + rounded_centroid[2] - radius] = label
    return array


def get_image_bounding_box(image):
    shape = image.shape
    return np.array(((0,shape[0]),(0,shape[1]),(0,shape[2])))


def is_label_enclosed_in_image(image_bounding_box, rounded_3d_centroid, radius):
    label_bounding_box = get_bounding_box_from_centroid_and_radius(rounded_3d_centroid, radius)
    protrusions = np.ones(label_bounding_box.shape, dtype=bool)
    protrusions[:,0] = label_bounding_box[:,0] < image_bounding_box[:,0]
    protrusions[:,1] = label_bounding_box[:,1] > image_bounding_box[:,1]
    label_protrudes = protrusions.any()
    label_is_enclosed_in_image = not label_protrudes
    return label_is_enclosed_in_image

def get_bounding_box_from_centroid_and_radius(rounded_3d_centroid, radius):
    bounding_box = np.zeros((3,2), dtype=int)
    rounded_3d_centroid = np.round(rounded_3d_centroid).astype(np.int16)
    radius = int(round(radius))
    bounding_box[:,0] = rounded_3d_centroid - radius
    bounding_box[:,1] = rounded_3d_centroid + radius
    #bounding_box[bounding_box < 0] = 0
    return bounding_box


def extract_box_of_radius(image, rounded_centroid, radius):
    bbox = get_bounding_box_from_centroid_and_radius(rounded_centroid, radius)
    bbox = shrink_cubic_bbox_to_fit_image(bbox, radius, image.shape)
    sub_image = image[bbox[0,0]:bbox[0,1],
                      bbox[1,0]:bbox[1,1],
                      bbox[2,0]:bbox[2,1]]
    return sub_image

def shrink_cubic_bbox_to_fit_image(bbox, radius, image_shape):
    image_bounding_box = np.array(((0,image_shape[0]),(0,image_shape[1]),(0,image_shape[2])))
    zeros = np.zeros(bbox.shape)
    protrusions = np.copy(zeros)
    protrusions[:,0] = image_bounding_box[:,0] - bbox[:,0]
    protrusions[:,1] = bbox[:,1] - image_bounding_box[:,1]
    #whereever protrusion is larger than zero, the bounding box is protruding from the image
    protrusions = np.maximum(protrusions, zeros)
    #next we calculate for each dimension how much the bbox protrudes from the image (sum)
    #and we take the maximum value of all. This by how much we need to reduce the box size at least
    correction = int(protrusions.sum(axis=1).max())
    bbox[:,0] += correction
    bbox[:,1] -= correction
    return bbox

def extract_extended_box(image, bbox, extension):
    extended_bbox = get_extended_bbox(bbox, extension)
    sub_image = extract_box(image, extended_bbox)
    return sub_image


def get_extended_bbox(bbox, extension):
    bbox = bbox.astype(np.int16)
    extended_bbox = np.zeros((bbox.shape), dtype=int)
    for i in range(3):
        for j, sign in enumerate((-1,1)):
            extended_bbox[i,j] = bbox[i,j] + sign * extension
    return extended_bbox


def get_center_of_bbox(bbox):
    """get the center of a bbox (shape 3,2) in the box coordinate system,
    not in the image coordinate system
    """
    center = (bbox[:,1] - bbox[:,0]) // 2
    center = center.astype(np.int16)
    return center


def extract_box(image, bbox):
    bbox = bbox.astype(np.int16)
    clipped_bbox = clip_box_to_image_size(image, bbox)
    sub_image = image[clipped_bbox[0,0]:clipped_bbox[0,1],
                      clipped_bbox[1,0]:clipped_bbox[1,1],
                      clipped_bbox[2,0]:clipped_bbox[2,1]]
    return sub_image


def clip_box_to_image_size(image, bbox):
    clipped_bbox = np.zeros(bbox.shape, dtype=int)
    clipped_bbox[:, 0] = np.maximum(0, bbox[:, 0])
    clipped_bbox[:, 1] = np.minimum(image.shape, bbox[:, 1])
    return clipped_bbox

def crop_edges(image, radius):
    """sets all pixels at a distance lower than radius from the edge of the image to 0
    """
    mask = np.ones_like(image)
    radius = int(np.ceil(radius))
    mask[:radius,] = 0
    mask[-radius:,] = 0
    mask[:,:radius] = 0
    mask[:,-radius:] = 0
    mask[:,:,:radius] = 0
    mask[:,:,-radius:] = 0
    return mask * image

def get_label_largest_radius(bbox):
    radii = get_label_radii(bbox)
    largest_radius = radii.max()
    return largest_radius


def get_label_radii(bbox):
    #this function is equivalent to get_center_of_bbox
    radii = (bbox[:,1] - bbox[:,0]) // 2
    radii = radii.astype(np.int16)
    return radii

def main():
    base_dir = Path().absolute()
    base_dir = base_dir / 'data'
    dataset_lst = [e for e in base_dir.iterdir() if e.is_dir()]
    error_str = f"Error: please chose a number between 0 and and {len(dataset_lst) - 1}, or q"
    while True:
        print(f"""Which dataset are we working on? Please enter a number between 0 and {len(dataset_lst) - 1},"
              or q to quit""")
        # [print(x) for x.name in dataset_lst]
        for i, p in enumerate(dataset_lst):
            print(f"{i}: {p.name}")
        print("?")
        choice = input()
        if choice == 'q':
            print("exiting")
            sys.exit(0)
        if choice.isdigit():
            if int(choice) in range(len(dataset_lst)):
                break
        print(error_str)
    dataset_dir = dataset_lst[int(choice)]
    run_default_pipeline(dataset_dir)

def embed_array_in_array(small_array, large_array, start_coordinates=(0,0,0)):
    large_array = large_array.copy()
    s0,s1,s2 = small_array.shape
    i,j,k = start_coordinates
    large_array[i:i+s0,j:j+s1,k:k+s2] = small_array
    return large_array

def embed_array_in_array_centered(small_array, large_array):
    start_coordinates = ((np.array(large_array.shape) - np.array(small_array.shape))//2).astype(np.int16)
    return embed_array_in_array(small_array, large_array, start_coordinates=start_coordinates)

def get_radial_profile(image, origin=None):
    """get radial profile. If origin is None, then origin is set to
    the center of the image
    """
    if origin is None:
        origin = np.array(image.shape)//2
    z, y, x = np.indices((image.shape))
    r = np.sqrt((x - origin[0])**2 + (y - origin[1])**2 + (z - origin[2])**2)
    r = r.astype(np.int16)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    return radial_profile

def get_3d_radial_average_from_profile(radial_profile,image_shape):
    #we do not need to care about what the origin of the radial profile was.
    #we always want the 3d radial average to have its origin centered. and we
    #want it to have the same image shape as the image that gave rise to the
    #radial profile.
    a, b, c = [s//2 for s in image_shape]
    z, y, x = np.mgrid[-a:a, -b:b, -c:c]
    rback = np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(np.int16)
    radial_average_3d = radial_profile[rback]
    return radial_average_3d

def get_3d_radial_average(image,origin=None):
    """
    get 3d radial average from image
    :param image: image to average
    :param origin: origin for rotational average. if None, then it will be set to the center of the image.
    :return: rotationally averaged image
    """
    radial_profile = get_radial_profile(image, origin)
    radial_average_3d = get_3d_radial_average_from_profile(radial_profile,image.shape)
    return radial_average_3d

def get_optimal_sphere_radius_from_radial_profile(radial_profile,tight=False):
    """
    get the radius that includes all the membrane density
    """
    i_membrane_center, _ = get_sphere_membrane_center_and_density_from_radial_profile(radial_profile)
    i_upper_limit = get_radial_profile_i_upper_limit(radial_profile)
    i_membrane_outer_halo = i_membrane_center + radial_profile[i_membrane_center:i_upper_limit].argmax()
    derivative2 = np.diff(radial_profile,2)
    if tight:
        tightened_derivative2 = ndimage.gaussian_filter1d(derivative2, 2)[i_membrane_center:i_membrane_outer_halo].argmin() + i_membrane_center
        optimal_radius = i_membrane_center + ndimage.gaussian_filter1d(derivative2, 1)[i_membrane_center:tightened_derivative2-1].argmin()
        # optimal_radius = i_membrane_center + ndimage.gaussian_filter1d(derivative2, 1)[i_membrane_center:i_membrane_outer_halo].argmin()
    else:
        # optimal_radius = i_membrane_center + ndimage.gaussian_filter1d(derivative2, 1)[i_membrane_center:i_membrane_outer_halo+1].argmin()
        optimal_radius = i_membrane_center + ndimage.gaussian_filter1d(derivative2, 1)[i_membrane_center:i_membrane_outer_halo].argmin()
    return(optimal_radius)


def get_sphere_membrane_center_and_density_from_radial_profile(radial_profile):
    i_lower_limit = round(len(radial_profile)*0.1)
    i_upper_limit = get_radial_profile_i_upper_limit(radial_profile)
    i_membrane_center = i_lower_limit + radial_profile[i_lower_limit:i_upper_limit].argmin()
    density = radial_profile[i_membrane_center]
    return i_membrane_center, density


def get_radial_profile_i_upper_limit(radial_profile):
    length = len(radial_profile)
    i_upper_limit = round(length * 0.75)
    i_upper_limit = round(length * 0.70)
    return i_upper_limit


def get_sphere_membrane_thickness_and_density_from_radial_profile(radial_profile):
    i_membrane_center, density = get_sphere_membrane_center_and_density_from_radial_profile(radial_profile)
    sphere_radius = get_optimal_sphere_radius_from_radial_profile(radial_profile)
    thickness = 2 * (sphere_radius - i_membrane_center)
    return thickness, density

def get_sphere_membrane_thickness_and_density_from_image(image):
    radial_profile = get_radial_profile(image)
    thickness, density = get_sphere_membrane_thickness_and_density_from_radial_profile(radial_profile)
    d = interpolate.interp1d(np.arange(len(radial_profile)),radial_profile)
    xnew = np.arange(0, len(radial_profile)-1, (len(radial_profile)-0.9999999999) / 100)
    ynew = d(xnew)
    return thickness, density,ynew

def get_optimal_sphere_radius_from_image(image, tight=False):
    radial_profile = get_radial_profile(image)
    optimal_radius = get_optimal_sphere_radius_from_radial_profile(radial_profile,tight=tight)
    return optimal_radius

def get_shift_between_images(reference_image, moving_image):
    try:
        shift, _, _ = skimage.registration.phase_cross_correlation(reference_image, moving_image,normalization=None)
    except ValueError:
        shift = np.zeros(3)
        print("get_shift_between_images failed, shift set to 0,0,0")
    return shift

def get_shift_of_sphere(image,origin=None):
    average_image = get_3d_radial_average(image, origin)
    shift = get_shift_between_images(average_image, image)
    return shift

def get_optimal_sphere_position_and_radius(image, rounded_centroid, radius, margin=5, max_cycles=10, max_shift_ratio=0.5, tight=False):
    image_box = extract_box_of_radius(image, rounded_centroid, radius + margin)
    max_shift = max_shift_ratio * np.linalg.norm(image.shape)
    total_shift = np.array((0,0,0))
    no_change_count = 0
    for i in range(max_cycles):
        shift = get_shift_of_sphere(image_box)
        total_shift = total_shift + shift
        if np.linalg.norm(total_shift) > max_shift:
            total_shift -= shift
            break
        new_centroid = (rounded_centroid - total_shift).astype(np.int16)
        if np.all(shift == np.zeros(3)):
            no_change_count += 1
        if no_change_count > 1:
            break
        image_box = extract_box_of_radius(image, new_centroid, radius + margin)
        new_radius = get_optimal_sphere_radius_from_image(image_box,tight=tight)
        if new_radius != radius:
            radius = new_radius
            image_box = extract_box_of_radius(image, new_centroid, radius + margin)
    return total_shift, radius

def rearrange_labels(image_label, dtype=np.int16):
    """rearrange the labels so that they go from 1 to n labels. Label order is preserved. The goal
    is to avoid any empty label"""
    uniques = np.unique(image_label)
    replace = {v: i for i, v in enumerate(uniques)}
    lookup_table = np.arange(0,uniques[-1]+1).astype(dtype)
    lookup_table[[*replace.keys()]] = [*replace.values()]
    return lookup_table[image_label]


def run_default_pipeline(dataset_dir,force=True,scale_proportion=None,within_cytoplasm=False):
    dataset_dir = Path(dataset_dir)
    myPipeline = pipeline.Pipeline(dataset_dir)
    myPipeline.setup_cryovesnet_dir(make_masks=within_cytoplasm)
    # myPipeline.evaluation()
    myPipeline.run_deep(force_run=force,rescale=scale_proportion)
    myPipeline.rescale(force_run=force)
    myPipeline.label_vesicles(within_segmentation_region =within_cytoplasm)
    myPipeline.label_vesicles_adaptive(separating=True)
    # myPipeline.threshold_tuner()
    # myPipeline.label_convexer()
    myPipeline.make_spheres()
    myPipeline.repair_spheres()
    #myPipeline.make_full_modfile()
    #myPipeline.make_full_label_file()

    # myPipeline.remove_small_labels()
    # myPipeline.make_full_modfile()
    # myPipeline.make_full_label_file()
    # myPipeline.initialize_pyto()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()