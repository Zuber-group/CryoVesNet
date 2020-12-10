import sys
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import numpy as np
import pandas as pd
import os
import mrcfile
from scipy import ndimage
from tqdm import tqdm
import pipeline
from pathlib import Path




def min_volume_of_vesicle(path_to_file, radius_thr = 12):
    """
    :param path_to_file:  path to tomogram
    :param radius_thr: minimum radius of vesicle in nanometer
    :return: minimum volume of vesicle in voxels
    """
    # read voxel size of tomogram
    with mrcfile.open(path_to_file, header_only = True) as tomo:
        voxel_size = float(tomo.voxel_size.x)
    volume_of_vesicle = (4.0 / 3.0) * np.pi * (radius_thr / voxel_size * 10) ** 3
    return volume_of_vesicle


def save_label_to_mrc(labels,path_to_file,template_path=None):
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
            mrc.header['origin'].x = template.header['origin'].x * -1
            mrc.header['origin'].y = template.header['origin'].y * -1
            mrc.header['origin'].z = template.header['origin'].z * -1
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

def find_threshold_per_vesicle_intensity(int_image, image_label, mask_image, dilation, convex, minimum_volume_of_vesicle):
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
    vesicle_regions = skimage.measure.regionprops_table(image_label, properties=('label', 'bbox'))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    #BZ: I don't think there is any reason to extend the box for this function
    #I left the option to do it (just change extension to whatever desired)
    box_extension = 0
    corrected_labels = np.zeros(image_label.shape)
    for i in tqdm(range(len(labels))):
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


def make_vesicles_spherical(image_label, diOrEr=0):
    corrected_labels = np.zeros(image_label.shape, dtype=int)
    image_bounding_box = get_image_bounding_box(image_label)
    vesicle_regions = pd.DataFrame(skimage.measure.regionprops_table(image_label,
                                                                     properties=('centroid', 'label', 'bbox')))
    bboxes = get_bboxes_from_regions(vesicle_regions)
    centroids = get_centroids_from_regions(vesicle_regions)
    labels = get_labels_from_regions(vesicle_regions)
    for i in range(len(vesicle_regions)):
        radius = get_label_largest_radius(bboxes[i] , diOrEr) #this is an integer
        rounded_centroid = np.round(centroids[i]).astype(np.int) #this is an array of integers
        label = labels[i]
        label_box = extract_box_of_radius(image_label, rounded_centroid, radius)
        if is_label_enclosed_in_image(image_bounding_box, rounded_centroid, radius):
            corrected_labels = put_spherical_label_in_array(corrected_labels, rounded_centroid, radius, label)
        else:
            corrected_labels = put_original_label_in_array(corrected_labels, label_box, label, rounded_centroid, radius)
    corrected_labels = skimage.morphology.label(corrected_labels >= 1, connectivity=1)
    corrected_labels = corrected_labels.astype(np.uint16)
    corrected_labels = oneToOneCorrection(image_label, corrected_labels)
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


def put_spherical_label_in_array(array, rounded_centroid, radius, label):
    sphere = skimage.morphology.ball(radius)
    px, py, pz = np.where(sphere)
    array[px + rounded_centroid[0] - radius,
          py + rounded_centroid[1] - radius,
          pz + rounded_centroid[2] - radius] = label
    return array


def put_original_label_in_array(array, label_box, label, rounded_centroid, radius):
    label_mask = label_box == label
    px, py, pz = np.where(label_mask)
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
    bounding_box[:,0] = rounded_3d_centroid - radius
    bounding_box[:,1] = rounded_3d_centroid + radius
    return bounding_box


def extract_box_of_radius(image, rounded_centroid, radius):
    bbox = get_bounding_box_from_centroid_and_radius(rounded_centroid, radius)
    sub_image = image[bbox[0,0]:bbox[0,1],
                      bbox[1,0]:bbox[1,1],
                      bbox[2,0]:bbox[2,1]]
    return sub_image


def extract_extended_box(image, bbox, extension):
    extended_bbox = get_extended_bbox(bbox, extension)
    sub_image = extract_box(image, extended_bbox)
    return sub_image


def get_extended_bbox(bbox, extension):
    bbox = bbox.astype(int)
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
    center = center.astype(int)
    return center


def extract_box(image, bbox):
    bbox = bbox.astype(int)
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


def get_label_largest_radius(bbox, diOrEr):
    radii = get_label_radii(bbox)
    largest_radius = radii.max() + diOrEr
    return largest_radius


def get_label_radii(bbox):
    #this function is equivalent to get_center_of_bbox
    radii = (bbox[:,1] - bbox[:,0]) // 2
    radii = radii.astype(int)
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


def run_default_pipeline(dataset_dir):
    dataset_dir = Path(dataset_dir)
    myPipeline = pipeline.Pipeline(dataset_dir)
    myPipeline.setup_prepyto_dir()
    myPipeline.evaluation()
    # myPipeline.run_deep()
    # myPipeline.zoom()
    # myPipeline.label_vesicles()
    # myPipeline.threshold_tuner()
    # myPipeline.label_convexer()
    # myPipeline.sphere_vesicles()
    # myPipeline.remove_small_labels()
    # myPipeline.make_full_modfile()
    # myPipeline.make_full_label_file()
    # myPipeline.initialize_pyto()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
