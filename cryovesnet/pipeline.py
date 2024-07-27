import math
import os
import platform
import sys
from pathlib import Path
import shutil

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
import skimage
import cryovesnet
from . import pyto_scripts
from .unetmic.unetmic import segment as segseg
from . import weights
# import evaluation_class
from . import mrc_cleaner
from . import visualization
from . import evaluation_class
import numpy as np
import pandas as pd
import mrcfile
from tqdm import tqdm
from scipy.stats import chi2, pearsonr, spearmanr
from collections.abc import Iterable
import subprocess
import shutil
from colorama import init, Fore, Style

# Initialize colorama (required for Windows)
init()


try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ModuleNotFoundError:
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("pipeline: tensorflow is not installed, some function will not work.")

if platform.system() == 'Darwin': os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_raw(path_to_file):
    if os.path.splitext(os.path.split(path_to_file)[1])[1] == '.tif':
        image = skimage.io.imread(path_to_file)

    else:
        image = mrcfile.open(path_to_file)
        image = image.data

    return image


def convert_mrc_to_npy(mrc_file_path, output_path):
    # Open the MRC file and read the data
    with mrcfile.open(mrc_file_path, mode='r') as mrc:
        data = mrc.data

    # Prepare the output file path
    # npy_file_path = os.path.splitext(mrc_file_path)[0] + '.npy'

    # Save the data to a .npy file
    np.save(output_path, data)


class Pipeline():

    def __init__(self, path_to_folder, pattern='*.rec.nad'):
        # BZ replaced self.path_to_folder by self.dir
        self.dir = Path(path_to_folder).absolute()
        print(Style.BRIGHT + Fore.BLUE + f"CryoVesNet Pipeline: the pipeline is created for {self.dir}" + Style.RESET_ALL)
        # BZ to get a clearer overview of data structure, all directory paths are defined here
        self.deep_dir = self.dir / 'deep'
        self.save_dir = self.dir / 'cryovesnet'
        self.pyto_dir = self.dir / 'pyto'
        pattern = pattern
        # pattern = '*.rec'
        print(pattern)
        try:
            self.image_path = next(self.dir.glob(pattern))  # was earlier called path_to_file
        except StopIteration:
            print(f"Error: {self.dir} does not contain file matching the pattern {pattern}. Exiting")
            sys.exit(1)
        # self.binned_deep_mask_path = self.deep_dir/(self.image_path.stem + '_wreal_mask.tiff')
        # there are issues loading a 3D 64-bit tiff --> shape comes incorrect. Therefore we use the np array save on disk
        self.binned_deep_mask_path = self.deep_dir / (self.image_path.stem + '_segUnet.npy')
        self.deep_mask_path = self.save_dir / (self.image_path.stem + '_zoomed_mask.mrc')
        self.deep_winners_mask_path = self.save_dir / (self.image_path.stem + '_winning_rescale_factors.mrc')
        self.cytomask_path = self.save_dir / (self.image_path.stem + '_cytomask.mrc')
        self.active_zone_mask_path = self.save_dir / (self.image_path.stem + '_azmask.mrc')
        self.deep_labels_path = self.save_dir / (self.image_path.stem + '_deep_labels.mrc')
        self.clean_deep_labels_path = self.save_dir / (self.image_path.stem + '_clean_deep_labels.mrc')
        self.threshold_tuned_labels_path = self.save_dir / (self.image_path.stem + '_intensity_corrected_labels.mrc')
        self.convex_labels_path = self.save_dir / (self.image_path.stem + '_convex_labels.mrc')
        self.ellipsoid_labels_path = self.save_dir / (self.image_path.stem + '_ellipsoid_labels.mrc')
        self.mancorr_labels_path = self.save_dir / (self.image_path.stem + '_mancorr.mrc')
        self.sphere_labels_path = self.save_dir / (self.image_path.stem + '_sphere.mrc')
        self.sphere_df_path = self.save_dir / (self.image_path.stem + '_sphere_dataframe.pkl')
        self.no_small_labels_path = self.save_dir / (self.image_path.stem + '_no_small_labels.mrc')
        self.final_vesicle_labels_path = self.save_dir / (self.image_path.stem + '_final_vesicle_labels.mrc')
        self.full_labels_path = self.save_dir / 'labels.mrc'

        self.trash_df = None
        self.radials = None
        self.image = None  # placeholder for real image array
        self.binned_deep_mask = None  # placeholder for binned_deep_mask
        self.deep_mask = None  # placeholder for unet mask image array
        self.deep_winners_mask = None
        self.cytomask = None
        self.active_zone_mask = None
        self.clean_deep_labels = None
        self.deep_labels = None  # placeholder for cleaned label array
        self.threshold_tuned_labels = None  # placeholder for threshold tuned label array
        self.ellipsoid_labels = None
        self.convex_labels = None
        self.mancorr_labels = None
        self.sphere_labels = None
        self.no_small_labels = None
        self.final_vesicle_labels = None
        self.full_labels = None
        self.garbage_collector = {'image', 'binned_deep_mask', 'deep_mask', 'cytomask',
                                  'active_zone_mask', 'deep_labels','clean_deep_labels', 'ellipsoid_labels',
                                  'threshold_tuned_labels', 'convex_labels', 'mancorr_labels', 'sphere_labels',
                                  'no_small_labels', 'final_vesicle_labels'}
        self.vesicle_mod_path = self.save_dir / 'vesicles.mod'
        self.active_zone_mod_path = self.dir / 'az.mod'
        self.cell_outline_mod_path = self.dir / 'cell_outline.mod'
        self.full_mod_path = self.save_dir / 'full_cryovesnet.mod'
        with pkg_resources.path(weights, 'weights.h5') as p:
            self.unet_weight_path = p
        self.check_files()
        self.image_shape = mrcfile.utils.data_shape_from_header(mrcfile.open(self.image_path, header_only=True).header)
        self.voxel_size = cryovesnet.get_voxel_size_in_nm(self.image_path)
        self.min_radius = cryovesnet.min_radius_of_vesicle(self.voxel_size)
        self.min_vol = cryovesnet.min_volume_of_vesicle(self.voxel_size)
        print("Pixel size: ", self.voxel_size)
        if self.image_path.exists():
            self.set_array('image')
        # if self.sphere_df_path.exists():
        #     self.sphere_df = pd.read_pickle(self.sphere_df_path)

    def quick_setup(self, labels_to_load=['sphere_labels']):
        self.set_array('image')
        for label_name in labels_to_load:
            self.set_array(label_name)

    def set_array(self, array_name, datatype=None):
        """
        checks if self.array_name contains data. If not, sets self.array_name to an array corresponding
        to the MRC file specified in self.array_name_path
        example: self.set_array("image") will set self.image to the array corresponding to the MRC file
        with path self.image_path
        """
        if array_name == 'last_output_array_name':
            array_name = self.last_output_array_name
        path_name = array_name + "_path"
        my_path = getattr(self, path_name)
        if getattr(self, array_name) is None:
            my_array = mrcfile.open(my_path).data
            setattr(self, array_name, my_array)
        if datatype is not None:
            # warnings.warn("Pipeline.set_array. When changing datatype, data is currently not rescaled, potentially leading to unexpected results.")
            setattr(self, array_name, getattr(self, array_name).astype(datatype))
        return array_name

    def set_segmentation_region_from_mod(self, datatype=np.uint8, force_generate=False):
        if (not self.cytomask_path.exists()) or force_generate:
            cmd = f"imodmop -mode 6 -o 1 -mask 1 \"{self.cell_outline_mod_path}\" \"{self.image_path}\" \"{self.cytomask_path}\""
            # print(cmd)
            os.system(cmd)
        self.set_array('cytomask', datatype=datatype)

    def set_active_zone_array_from_mod(self, datatype=np.uint8, force_generate=False):
        if (not self.active_zone_mask_path.exists()) or force_generate:
            cmd = f"imodmop -mode 6 -o 1 -tube 1 -diam 3 -pl -mask 1 \"{self.active_zone_mod_path}\" \"{self.image_path}\" \"{self.active_zone_mask_path}\""
            os.system(cmd)
        self.set_array('active_zone_mask', datatype=datatype)

    def get_additional_non_vesicle_objects_from_mod(self, mod_objects, destination_labels, modfile_path=None,
                                                    datatype=np.uint16):
        if not modfile_path: modfile_path = self.full_mod_path
        temp_label_path = self.save_dir / 'temp_label.mrc'
        mod_objects = self.to_list(mod_objects)
        destination_labels = self.to_list(destination_labels)
        cmd = f"imodmop -mode 0 -objects {','.join(str(e) for e in mod_objects)} -labels {','.join(str(e) for e in destination_labels)}  \"{modfile_path}\" \"{self.image_path}\" \"{temp_label_path}\""
        os.system(cmd)
        with mrcfile.open(temp_label_path) as mrc:
            label_array = mrc.data.astype(datatype)
        temp_label_path.unlink()
        return label_array

    def get_vesicles_from_mod(self, mod_objects, modfile_path=None, datatype=np.uint16):
        if not modfile_path: modfile_path = self.full_mod_path
        temp_label_path = self.save_dir / 'temp_label.mrc'
        mod_objects = self.to_list(mod_objects)
        cmd = f"imodmop -mode 0 - mask 1 -3dscat -objects {','.join(str(e) for e in mod_objects)}  \"{modfile_path}\" \"{self.image_path}\" \"{temp_label_path}\""
        os.system(cmd)
        with mrcfile.open(temp_label_path) as mrc:
            label_array = mrc.data.astype(datatype)
        temp_label_path.unlink()
        label_array = skimage.morphology.label(label_array)
        return label_array

    def clear_memory(self, exclude=None):
        """
        set all object arrays to None in order to free up memory
        :param exclude: either a string or a collection of strings that define the object array(s) which should
        not be erased
        """
        garbage = self.garbage_collector.copy()
        if exclude:
            exclude = self.to_list(exclude)
            for ex in exclude:
                garbage.remove(ex)
        for g in garbage:
            setattr(self, g, None)

    def print_output_info(self):
        array_name = self.last_output_array_name
        path_name = array_name + "_path"
        my_path = getattr(self, path_name)
        print(f"last output array name: {array_name}")
        print(f"last mrc file saved : {my_path.relative_to(self.dir)}")

    def to_list(self, obj):
        if isinstance(obj, Iterable) and (type(obj) != str):
            my_list = list(obj)
        else:
            my_list = [obj]
        return my_list

    # def check_files(self, file_list=['dir', 'image_path','cell_outline_mod_path']):
    def check_files(self, file_list=['dir', 'image_path']):
        """
        Check if all input files as specified in file_list exist
        """
        # file_list = ['dir', 'image_path', 'active_zone_mod_path','cell_outline_mod_path']
        missing = []
        for p_str in file_list:
            p = getattr(self, p_str)
            if not p.exists(): missing.append(str(p))
        if len(missing):
            err = f"""the following file(s) and/or directory(s) are missing: {", ".join(missing)}"""
            raise IOError(err)

    def prepare_deep(self, erase_existing=False):
        self.deep_dir.mkdir(exist_ok=True)
        if erase_existing:
            for p in self.deep_dir.glob("*"):
                p.unlink()
        tf.get_logger().setLevel('INFO')
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        tf.keras.backend.set_session(tf.Session(config=config))
        # self.network_size = 64

    def run_membrain(self, path_to_model= '/home/amin/membrain-seg/checkpoints/checkpoints/membrain-seg_v0_1-epoch=999-val_loss=0.52.ckpt', force_run=False):
        # We call the bash script ./membrain_segment_link.sh path_to_tomogram path_to_model output_dir


        # load image
        image = load_raw(self.image_path)
        rescale = self.voxel_size * 10 / 22.40
        self.prepare_deep(erase_existing=force_run)
        scaled_image_path = os.path.normpath(self.deep_dir) + '/' + \
                            os.path.splitext(os.path.split(self.image_path)[1])[0] + '_processed.mrc'
        if not os.path.exists(scaled_image_path) or force_run:
            # rescale
            image = skimage.transform.rescale(image, scale=rescale, preserve_range=True).astype(np.int16)
            with mrcfile.new(scaled_image_path, overwrite=True) as mrc:
                mrc.set_data(image.astype(np.int16))
        if os.path.exists(self.binned_deep_mask_path) and not force_run:
            return
        result = subprocess.call(f"/home/amin/PycharmProjects/CryoVesNetNEW/notebooks/membrain_segment_link.sh {scaled_image_path} {path_to_model} {self.deep_dir}".split())

        if result != 0:
            raise ValueError("Membrain failed to segment the image")

        # use shutil to move  self.deep_dir / (self.image_path.stem + '_scores.mrc'):

        # shutil.move(scaled_image_path[:-4]+ '_scores.mrc', self.binned_deep_mask_path)

        convert_mrc_to_npy(scaled_image_path[:-4]+ '_scores.mrc',self.binned_deep_mask_path)






    def run_deep(self, force_run=False, rescale=None, gauss=True, augmentation_level=1, weight_path=None):
        """
        Merged vesicle_segmentation and run_deep to make it a pipeline method
        all output files are saved in self.deep_dir
        """
        if rescale==None:
            rescale=self.voxel_size*10/22 #in case you use the pre-trained model
            # rescale = self.voxel_size * 10 / 14.69  # in case you use the pre-trained model

        if weight_path is None:
            weight_path = str(self.unet_weight_path.absolute())

        print("Rescale Factor: ",rescale)
        self.prepare_deep(erase_existing=force_run)
        if self.deep_dir.exists() and len(list(self.deep_dir.glob('*'))) >= 4 and not force_run:
            return
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet pipeline: Running unet segmentation if there are less than 4 file in ./deep directory" + Style.RESET_ALL)
        # segseg.full_segmentation(weight_path, self.image_path, self.deep_dir, rescale=rescale, gauss=True)
        segseg.full_segmentation(weight_path, self.image_path, self.deep_dir, rescale=rescale, gauss=gauss, augmentation_level=augmentation_level)



    def run_deep_at_multiple_rescale(self, max_voxel_size=3.14, min_voxel_size=1.57, nsteps=8):
        self.set_array('image')
        min_rescale = self.voxel_size / max_voxel_size
        max_rescale = self.voxel_size / min_voxel_size
        max_deep_mask = np.zeros_like(self.image, dtype=np.float16)
        deep_winners_mask = np.zeros_like(self.image, dtype=np.float16)
        for rescale in tqdm(np.linspace(min_rescale, max_rescale, num=nsteps, endpoint=True)):
            print(Style.BRIGHT + Fore.BLUE + f"CryoVesNet pipeline: run_deep_at_multiple_rescale - rescale = {rescale}" + Style.RESET_ALL)
            self.run_deep(force_run=True, rescale=rescale)
            self.zoom(force_run=True)
            larger_mask = self.deep_mask > max_deep_mask
            max_deep_mask[larger_mask] = self.deep_mask[larger_mask]
            deep_winners_mask[larger_mask] = rescale
        self.deep_mask = max_deep_mask
        cryovesnet.save_label_to_mrc(self.deep_mask, self.deep_mask_path, template_path=self.image_path)
        self.deep_winners_mask = deep_winners_mask
        cryovesnet.save_label_to_mrc(self.deep_winners_mask, self.deep_winners_mask_path, template_path=self.image_path)

    def setup_cryovesnet_dir(self, make_masks=True, initialize=False, memkill=True):
        print(Style.BRIGHT + Fore.BLUE +"CryoVesNet Pipeline: setting up cryovesnet directory" + Style.RESET_ALL)
        self.save_dir.mkdir(exist_ok=True)
        if initialize:
            self.deep_labels = np.zeros(self.image_shape, dtype=np.uint16)
            cryovesnet.save_label_to_mrc(self.deep_labels, self.deep_labels_path, template_path=self.image_path)
            self.last_output_array_name = "deep_labels"
        print(self.save_dir)
        if make_masks:
            #chech if the cell_outline_mod_path exists
            if not self.cell_outline_mod_path.exists():
                raise ValueError("cell_outline_mod_path does not exist, make sure you have cell_outline.mod in the directory or specify make_masks=False in the setup_cryovesnet_dir")
            else:
                self.set_segmentation_region_from_mod()
            #check if the active_zone_mod_path exists
            if not self.active_zone_mod_path.exists():
                print("active_zone_mod_path does not exist!")
            else:
                self.set_active_zone_array_from_mod()
        if memkill:
            self.clear_memory()

    def zoom(self, force_run=False, slice_range=None,memkill=True):
        """
        Zoom the deep mask
        :param memkill:
        :return:
        """
        print(Style.BRIGHT + Fore.BLUE +"CryoVesNet Pipeline: zooming the unet mask" + Style.RESET_ALL)
        self.last_output_array_name = 'deep_mask'
        if self.deep_mask_path.exists() and not force_run:
            print("Skipping because a full sized deep mask is already saved on the disk.")
            print("If you want to force the program to make a new full sized deep mask, set force_run to True.")
            return
        self.binned_deep_mask = np.load(self.binned_deep_mask_path)
        self.set_array('image')
        self.deep_mask = skimage.transform.resize(self.binned_deep_mask, output_shape=np.shape(self.image),
                                                  preserve_range=True).astype(np.float32)
        # set the z slice under min_slice and over max_slice to zero in deep_mask
        print(self.image_shape)
        if slice_range is None:
            self.min_slice= 0
            self.max_slice = self.image_shape[0]
        else:
            self.min_slice  = slice_range[0]
            self.max_slice = slice_range[1]

        self.deep_mask[:self.min_slice,:,:] = 0
        self.deep_mask[self.max_slice:,:,:] = 0



        cryovesnet.save_label_to_mrc(self.deep_mask, self.deep_mask_path, template_path=self.image_path)
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def label_vesicles_simply(self, input_array_name='last_output_array_name', threshold_coef=1.0, expanding=False,
                              convex=False, separating =False,  memkill=True):
        # threshold_coef=0.986

        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: running label_vesicles_simply" + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name

        self.set_array('image')
        self.set_array('deep_mask')
        self.set_array(input_array_name)

        # if input_array_name == 'deep_mask':
        #     self.deep_mask = getattr(self, input_array_name)
        #     self.deep_mask = cryovesnet.crop_edges(self.deep_mask, radius=self.min_radius)
        # if within_segmentation_region:
        #     self.outcell_remover(input_array_name='deep_mask', output_array_name='deep_mask', memkill=False)
        # #
        if input_array_name== 'deep_mask':
            self.deep_mask = getattr(self, input_array_name)
            opt_th, mean_shell_val = cryovesnet.my_threshold(self.image, self.deep_mask, min_thr=0.8,max_thr=1, step=0.01)
            self.deep_mask = cryovesnet.crop_edges(self.deep_mask, radius=self.min_radius)

            threshold = threshold_coef * opt_th
            self.threshold_tuned_labels = threshold
            print(threshold)
            #
            deep_labels = skimage.morphology.label(self.deep_mask > threshold)
            self.deep_labels = deep_labels.astype(np.uint16)
            # deep_labels , _ = cryovesnet.find_threshold_per_vesicle_intensity(self.image, self.deep_labels , self.deep_mask, self.min_vol)
        else:
            deep_labels = self.deep_labels

        # deep_labels= self.deep_labels
        # ves_table2 = cryovesnet.vesicles_table(deep_labels)
        # deep_labels, small_labels = cryovesnet.expand_small_labels(self.deep_mask, deep_labels, threshold, self.min_vol,p=1, q=4, t=0.8)


        if separating:
            ves_table = cryovesnet.vesicles_table(deep_labels)
            deep_labels = cryovesnet.collision_solver(self.deep_mask, deep_labels, ves_table, self.threshold_tuned_labels , delta_size=10)

        if expanding:
            deep_labels, small_labels = cryovesnet.expand_small_labels(self.deep_mask, deep_labels, self.threshold_tuned_labels, self.min_vol, p=1, q=4, t=0.8)

            if len(small_labels):
                print(
                    "The following labels are too small and couldn't be expanded with decreasing deep mask threshold. Therefore they were removed.")
                print("You may want to inspect the region of their centroid, as they may correspond to missed vesicles.")
                print(small_labels)
                deep_labels[np.isin(deep_labels, small_labels.index)] = 0

        if convex==True:
            deep_labels = cryovesnet.fast_pacman_killer(deep_labels)
            # deep_labels = cryovesnet.pacman_killer(deep_labels)

        ves_table = cryovesnet.vesicles_table(deep_labels)
        deep_labels,badVesicles = cryovesnet.remove_outliers(deep_labels, ves_table, self.min_vol)

        ves_table = cryovesnet.vesicles_table(deep_labels)

        self.clean_deep_labels = deep_labels.astype(np.uint16)
        cryovesnet.save_label_to_mrc(self.clean_deep_labels, self.clean_deep_labels_path, template_path=self.image_path)
        # free up memory
        self.last_output_array_name = 'clean_deep_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()
        return ves_table,badVesicles

    def label_vesicles(self, input_array_name='last_output_array_name', within_segmentation_region=False, threshold_coef=None,
                       memkill=False):
        """label vesicles from the zoomed deep mask
        :param input_array_name: name of the array to be labelled
        :param within_bound: restrict labelling to segmentation_region? (currently called cytomask)
        this is done by setting every voxel outside the bounding volume to 0 in deep_mask ("_zoomed_mask.mrc)
        :param memkill:
        :return:
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: running label_vesicles" + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name

        self.set_array('image')
        self.set_array(input_array_name)
        self.deep_mask = getattr(self, input_array_name)
        if within_segmentation_region:
            if self.cytomask_path.exists():
                self.set_array('cytomask')
                self.outcell_remover(input_array_name='deep_mask', output_array_name='deep_mask', memkill=False)
            else:
                print(Style.BRIGHT +
                    Fore.YELLOW + "Warning: cytomask is not set. Skipping outcell removal. If you do not want to remove outcell vesicles, set within_segmentation_region to False." + Style.RESET_ALL)

        if threshold_coef is None:
            opt_th, _ = segseg.find_threshold(self.image, self.deep_mask,min_th=0.8)
        else:
            opt_th = threshold_coef
        self.threshold_tuned_labels = opt_th
        print("Threshold: ", opt_th)

        image_label_opt = skimage.morphology.label(self.deep_mask > opt_th,connectivity=1)
        deep_labels = image_label_opt
        # _, deep_labels = segseg.mask_clean_up(image_label_opt)
        self.deep_labels = deep_labels.astype(np.uint16)
        cryovesnet.save_label_to_mrc(self.deep_labels, self.deep_labels_path, template_path=self.image_path)
        # free up memory
        self.last_output_array_name = 'deep_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image',
                                  'threshold_tuned_labels','deep_mask'])
        self.print_output_info()

    def mask_loader(self):
        max_mask = mrcfile.open(self.deep_winners_mask_path)
        self.deep_mask = max_mask.data
        cryovesnet.save_label_to_mrc(self.deep_mask, self.deep_mask_path, template_path=self.image_path)

    def outcell_remover(self, input_array_name='last_output_array_name',
                        output_array_name='deep_labels', memkill=True, force_generate=False):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: restricting labels to segmentation region" + Style.RESET_ALL)
        self.set_segmentation_region_from_mod()
        self.set_array(input_array_name)
        bounded_array = getattr(self, input_array_name) * self.cytomask
        setattr(self, output_array_name, bounded_array)
        output_path_name = output_array_name + '_path'
        cryovesnet.save_label_to_mrc(bounded_array, getattr(self, output_path_name), template_path=self.image_path)
        # free up memory
        if memkill:
            self.clear_memory(exclude=[output_array_name, 'image'])
        self.last_output_array_name = output_array_name
        self.print_output_info()

    def threshold_tuner(self, input_array_name='last_output_array_name', memkill=True):
        print(Style.BRIGHT + Fore.BLUE + 'CryoVesNet Pipeline: running threshold_tuner' + Style.RESET_ALL)
        self.set_array('image')
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.set_array('deep_mask')
        self.threshold_tuned_labels, mean_shell_arg = cryovesnet.find_threshold_per_vesicle_intensity(int_image=self.image,
                                                                                                      image_label=getattr(
                                                                                                       self,
                                                                                                       input_array_name),
                                                                                                      mask_image=self.deep_mask,
                                                                                                      dilation=0,
                                                                                                      convex=0,
                                                                                                      minimum_volume_of_vesicle=self.min_vol)
        self.threshold_tuned_labels = cryovesnet.oneToOneCorrection(getattr(self, input_array_name),
                                                                    self.threshold_tuned_labels)
        cryovesnet.save_label_to_mrc(self.threshold_tuned_labels, self.threshold_tuned_labels_path,
                                     template_path=self.image_path)
        # free up memory
        self.last_output_array_name = 'threshold_tuned_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def label_convexer(self, input_array_name='last_output_array_name', memkill=True):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: making vesicles convex" + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.convex_labels = cryovesnet.pacman_killer(getattr(self, input_array_name))
        cryovesnet.save_label_to_mrc(self.convex_labels, self.convex_labels_path, template_path=self.image_path)
        self.last_output_array_name = 'convex_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def interactive_cleaner(self, input_array_name='last_output_array_name', memkill=True):
        """
        starts interactive cleaner
        :param
        :return:
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: interactive cleaning. To continue close the napari window." + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.mancorr_labels = mrc_cleaner.interactive_cleaner(self.image, getattr(self, input_array_name))
        cryovesnet.save_label_to_mrc(self.mancorr_labels, self.mancorr_labels_path, template_path=self.image_path)
        # free up memory
        self.last_output_array_name = 'mancorr_labels'
        if memkill:
            self.clear_memory(exclude=self.last_output_array_name)
        self.print_output_info()

    def compute_sphere_dataframe(self, input_array_name='last_output_array_name', tight=False,keep_ellipsoid=False):
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        sphere_df, radials , ellipsoid_label = cryovesnet.get_sphere_dataframe(self.image, getattr(self, input_array_name),margin=2, tight=tight,keep_ellipsoid=keep_ellipsoid)
        radials = np.array(radials)
        self.radials = radials
        mu = np.mean(radials, axis=0)
        corr_s = [spearmanr(x, mu)[0] for x in radials]
        sphere_df['corr'] = corr_s
        mahalanobis_series = cryovesnet.mahalanobis_distances(sphere_df.drop(['center', 'corr'], axis=1))
        sphere_df['mahalanobis'] = mahalanobis_series
        print(len(sphere_df))
        sphere_df['p'] = 1 - chi2.cdf(sphere_df['mahalanobis'], 2)
        sphere_df['radials'] = radials.tolist()
        # self.trash_df= sphere_df[sphere_df['mahalanobis'] >= m]
        # sphere_df = sphere_df[sphere_df['mahalanobis'] <m]
        # sphere_df = sphere_df[sphere_df['corr'] > 0.3]
        print(len(sphere_df))

        # sphere_df["radial"] = radials
        # mu = np.mean(radials,axis=0)
        # corr_s= [spearmanr(x, mu)[0] for x in sphere_df["radial"]]
        # corr_p = [pearsonr(x, mu)[0] for x in sphere_df["radial"]]
        # sphere_df["corr"] = corr_s
        # print(len(sphere_df))
        # sphere_df = sphere_df[sphere_df['corr'] > 0.3]
        # print(len(sphere_df))
        sphere_df.to_pickle(self.sphere_df_path)
        self.sphere_df = sphere_df
        return sphere_df,ellipsoid_label

    def make_spheres(self, input_array_name='last_output_array_name', tight= False, keep_ellipsoid = False ,memkill=True):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: Making vesicles spherical." + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        _ , ellipsoid_tags = self.compute_sphere_dataframe(input_array_name, tight= tight,keep_ellipsoid=keep_ellipsoid)
        self.sphere_labels = cryovesnet.make_vesicle_from_sphere_dataframe(getattr(self, input_array_name), self.sphere_df)
        if keep_ellipsoid == True and len(ellipsoid_tags) > 0:
            print("ellipsoid_label",ellipsoid_tags)
            # for on ellipsoid_label and check where in the deep_labels is eqaul to the ellipsoid_label and then put all these values in sphere_labels
            temp= getattr(self, input_array_name)
            ellipsoid_labels = np.where(np.isin(temp, ellipsoid_tags), temp, self.sphere_labels).astype(np.uint16)
            self.ellipsoid_labels= ellipsoid_labels
            cryovesnet.save_label_to_mrc(self.ellipsoid_labels, self.ellipsoid_labels_path, template_path=self.image_path)
            self.last_output_array_name = 'ellipsoid_labels'
            self.print_output_info()



        cryovesnet.save_label_to_mrc(self.sphere_labels, self.sphere_labels_path, template_path=self.image_path)
        self.last_output_array_name = 'sphere_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()
        return self.sphere_df

    def repair_spheres(self, input_array_name='last_output_array_name',  memkill=True, p= 0.3, m=4, r=0):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: Repairing vesicles." + Style.RESET_ALL)

        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        # deep_labels = self.deep_labels.copy()
        sphere_labels = self.sphere_labels.copy()

        sphere_df = pd.read_pickle(self.sphere_df_path)
        sphere_df['radius'] = sphere_df['radius'] + r
        sphere_df = sphere_df[sphere_df['mahalanobis'] < m]
        sphere_df = sphere_df[sphere_df['corr'] > p]
        sphere_labels = cryovesnet.make_vesicle_from_sphere_dataframe(sphere_labels, sphere_df)
        # if cytomask is available, remove surrounding labels
        if (self.cytomask_path.exists()):
            self.set_array('cytomask')
            surround_labels = cryovesnet.surround_remover(sphere_labels, self.cytomask, self.min_vol)
            sphere_df.drop(surround_labels)
        else:
            print(Style.BRIGHT + Fore.YELLOW + "Warning: cytomask is not set. Surrounding labels are not removed." + Style.RESET_ALL)



        temp = cryovesnet.adjacent_vesicles(sphere_df)
        edited = []
        # print(temp)
        while (len(temp) > 0):
            print("#############################################################################")
            print(Fore.GREEN + "Adjacent vesicles: "+ Style.RESET_ALL)
            print(temp)
            for x in temp:
                # print(x)
                # print("e",edited)
                if x[0] not in edited and x[1] not in edited:
                    # print(x)
                    p = x[0]
                    q = x[1]
                    r1 = sphere_df['radius'].iloc[p]
                    r2 = sphere_df['radius'].iloc[q]
                    c1 = sphere_df['center'].iloc[p]
                    c2 = sphere_df['center'].iloc[q]
                    r3 = r1 + r2
                    d = math.ceil(math.sqrt(sum((c1 - c2) ** 2)))
                    new_r1 = math.floor(d * (r1 / r3)) - 1
                    new_r2 = math.floor(d * (r2 / r3)) - 1
                    print(sphere_df.iloc[p])
                    sphere_df.iat[p, 2] = int(new_r1)
                    print(sphere_df.iloc[p])
                    sphere_df.iat[q, 2] = int(new_r2)
                    print(r1, new_r1, r2, new_r2, d)
                    edited.append(p)
                    edited.append(q)
                    # temp = np.delete(temp,np.where(temp==x))
                    # temp = np.delete(temp,np.where(temp==[q,p]))
            temp = cryovesnet.adjacent_vesicles(sphere_df)
            edited = []

        sphere_df = sphere_df[sphere_df['radius'] > self.min_radius]
        # self.convex_labels = temp_best_corrected_labels
        self.convex_labels = cryovesnet.make_vesicle_from_sphere_dataframe(sphere_labels, sphere_df)
        print("Number of vesicles: ", len(sphere_df))
        cryovesnet.save_label_to_mrc(self.convex_labels, self.convex_labels_path, template_path=self.image_path)
        self.last_output_array_name = 'convex_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()


    def identify_spheres_outliers(self, bins=50, min_mahalanobis_distance=2.0):
        ax = self.sphere_df.mahalanobis.hist(bins=bins, color='blue')
        ylim0, ylim1 = ax.get_ylim()
        ax.vlines(min_mahalanobis_distance, ylim0, ylim1, linestyles='dashed', color='red',
                  label='Mahalanobis threshold for printing sphere details')
        ax.set_title("Outlier spheres detection")
        ax.set_xlabel("Mahalanobis")
        ax.set_ylabel("Number of spheres")
        print(self.sphere_df[self.sphere_df.mahalanobis > min_mahalanobis_distance].sort_values('mahalanobis'))
        print(
            "You should inspect the labels that have a high mahalanobis distance as they are the likeliest to be wrongly segmented")

    def fix_spheres_interactively(self, input_array_name='last_output_array_name', max_expected_diameter=50 ,memkill=True):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: fixing interactively" + Style.RESET_ALL)

        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array('image')
        self.set_array(input_array_name)
        visualization.add_points_modify_labels(self,max_diameter=max_expected_diameter)
        # visualization.add_points_remove_labels_v1(self)
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])

        self.print_output_info()

    def remove_small_labels(self, input_array_name='last_output_array_name', first_label=1, memkill=True):
        """
        remove any label smaller than self.volume_threshold_of_vesicle and renumber labels to avoid any empty label
        (starting from first_label)
        :param first_label: value of the first value to consider (typically 1 or 10, depending whether we have SV
        numbered from 1 or from 10)
        :param input_array_name:
        :param memkill:
        :return:
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: removing small labels" + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        label_index, label_size = np.unique(getattr(self, input_array_name), return_counts=True)
        sizes_labels = dict(zip(*(label_index, label_size)))
        small_labels = np.array([k for (k, v) in sizes_labels.items() \
                                 if (v < self.min_vol) and (k >= first_label)])
        selection = np.isin(getattr(self, input_array_name), small_labels)
        self.no_small_labels = getattr(self, input_array_name).copy()
        self.no_small_labels[selection] = 1  # set the small labels to 1, i.e. cytoplasm
        # remove gaps
        missing_labels = [i for i in range(first_label, max(label_index) + 1) if not i in label_index]
        small_labels = list(set(small_labels).union(missing_labels))
        small_labels.sort()
        for l in tqdm(small_labels[::-1], desc="reordering remaining labels."):
            self.no_small_labels = np.where(self.no_small_labels > l, self.no_small_labels - 1, self.no_small_labels)
        cryovesnet.save_label_to_mrc(self.no_small_labels, self.no_small_labels_path, template_path=self.image_path)
        # free up memory
        self.last_output_array_name = 'no_small_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def visualization_old_new(self, input_array_name1, input_array_name2, memkill=True):
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: visualizing two sets of labels. To continue, close napari window" + Style.RESET_ALL)
        self.set_array('image')
        self.set_array(input_array_name1)
        self.set_array(input_array_name2)
        visualization.viz_labels(self.image, [getattr(self, input_array_name1), getattr(self, input_array_name2)],
                                 ['Old', 'New'])
        if memkill:
            self.clear_memory(exclude=[input_array_name2, 'image'])

    def initialize_pyto(self, overwrite=False):
        """
        copy initial pyto to a project subfolder
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: setting up pyto folder" + Style.RESET_ALL)
        if self.pyto_dir.exists():
            if overwrite:
                self.pyto_dir.unlink()
            else:
                print(
                    f"{self.pyto_dir} exists. Skipping pyto file creation. To force reinitialization of pyto folder, set overwrite to True")
                return
        with pkg_resources.path(pyto_scripts, '.') as src:
            shutil.copytree(src, self.pyto_dir)

    def make_full_modfile(self, input_array_name='last_output_array_name', do_rearrange_labels=True, memkill=True, origin_adjustment=1):
        """
        convert the vesicle label file to a modfile and merge it with the initial modfile
        do_rearrange_labels: if True, then no empty label in the input labels array are left
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: making full mod file" + Style.RESET_ALL)
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        if do_rearrange_labels:
            self.final_vesicle_labels = cryovesnet.rearrange_labels(getattr(self, input_array_name))
            print("rearranged labels")
        else:
            self.final_vesicle_labels = getattr(self, input_array_name)
        cryovesnet.save_label_to_mrc(self.final_vesicle_labels, self.final_vesicle_labels_path,
                                     template_path=self.image_path, q=origin_adjustment)
        self.last_output_array_name = 'final_vesicle_labels'
        self.print_output_info()
        cmd0 = f"imodauto -f 3 -m 4 -h 0 -O 10 \"{self.final_vesicle_labels_path}\" \"{self.vesicle_mod_path}\""
        cmd1 = f"imodjoin -c \"{self.cell_outline_mod_path}\" \"{self.active_zone_mod_path}\" \"{self.vesicle_mod_path}\" \"{self.full_mod_path}\""
        os.system(cmd0)
        print("next:")
        os.system(cmd1)
        # os.system("imodjoin -c full_cryovesnet.mod vesicles.mod full_cryovesnet.mod")
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        print(f"full model file saved to {self.full_mod_path.relative_to(self.dir)}")

    def handpick(self):
        cmd0 = f"imod -W \"{self.image_path}\" \"{self.full_mod_path}\""
        os.system(cmd0)

    def make_full_label_file(self, vesicle_array_name='last_output_array_name',
                             include_segmentation_region=True, include_active_zone=True,
                             additional_non_vesicle_objects=[], additional_non_vesicle_destination_label=[],
                             additional_non_vesicle_object_modpath=None,
                             handpicked_vesicles_objects=[],
                             hand_picked_vesicles_modpath=None,
                             first_vesicle_index=10, memkill=True, origin_adjustment=1):
        """
        Assemble full label file from masks and full mod file
        :return:
        """
        print(Style.BRIGHT + Fore.BLUE + "CryoVesNet Pipeline: making full label file" + Style.RESET_ALL)
        self.full_labels = np.zeros(self.image_shape, dtype=np.uint16)
        if include_segmentation_region:
            self.set_segmentation_region_from_mod()
            self.full_labels[self.cytomask.nonzero()] = 1
            if memkill:
                self.cytomask = None
        if include_active_zone:
            self.set_active_zone_array_from_mod()
            self.full_labels[self.active_zone_mask.nonzero()] = 2
        if additional_non_vesicle_objects:
            if not additional_non_vesicle_destination_label:
                additional_non_vesicle_destination_label = additional_non_vesicle_objects.copy()
                additional_array = self.get_additional_non_vesicle_objects_from_mod(additional_non_vesicle_objects,
                                                                                    additional_non_vesicle_destination_label,
                                                                                    additional_non_vesicle_object_modpath)
                additional_indices = additional_array.nonzero()
                self.full_labels[additional_indices] = additional_array[additional_indices]
        actual_vesicle_array_name = self.set_array(vesicle_array_name)
        vesicle_label_array = skimage.morphology.label(getattr(self, actual_vesicle_array_name))
        vesicle_indices = vesicle_label_array.nonzero()
        offset = first_vesicle_index - 1
        self.full_labels[vesicle_indices] = vesicle_label_array[vesicle_indices] + offset
        if handpicked_vesicles_objects:
            handpicked_array = self.get_vesicles_from_mod(handpicked_vesicles_objects, hand_picked_vesicles_modpath)
            offset = self.full_labels.max()
            handpicked_indices = handpicked_array.nonzero()
            self.full_labels[handpicked_indices] = handpicked_array[handpicked_indices] + offset
        cryovesnet.save_label_to_mrc(self.full_labels, self.full_labels_path, template_path=self.image_path, q=origin_adjustment)
        if memkill:
            self.clear_memory()
        print(f"saved labels to {self.full_labels_path.relative_to(self.dir)}")

    def evaluation(self, reference_path=None, prediction_path=None):
        self.set_array("image")
        self.set_array("cytomask")
        # if input_array_name == 'last_output_array_name':
        #     input_array_name = self.last_output_array_name
        # self.set_array(input_array_name)
        # input_array_name = 'last_output_array_name'
        reference_path = mrc_cleaner.ask_file_path(self.dir, file_extension=('.mrc'))
        if prediction_path == None:
            prediction_path = mrc_cleaner.ask_file_path(self.save_dir, file_extension=('.mrc'))
            prediction = mrcfile.open(prediction_path)
        else:
            prediction_path = mrc_cleaner.ask_file_path(self.dir / prediction_path, file_extension=('.mrc'))
            prediction = mrcfile.open(prediction_path)
            # prediction = self.last_output_array

        reference = mrcfile.open(reference_path)
        # maskfile = mrcfile.open('./compare/labels_manual.mrc')
        # corrected = mrcfile.open('./compare/labels_automation.mrc')

        # real_image = umic.load_raw('./compare/Dummy_80.rec.nad')
        real_image = self.image
        mask = reference.data
        mask = mask * self.cytomask
        print(np.shape(mask))
        mask = mask >= 10
        corrected_labels = prediction.data

        corrected_labels_mask = corrected_labels >= 1
        evaluator = evaluation_class.ConfusionMatrix(corrected_labels_mask, mask)
        # print(evaluator)

        print("The Pixel Accuracy is: " + str(evaluator.accuracy()))
        print("The  Intersection-Over-Union is: " + str(evaluator.jaccard_index()))
        print("The Dice Metric: " + str(evaluator.dice_metric()))
        print("Former Dice: " + str(evaluator.former_dice()))
        visualization.viz_labels(self.image, [mask, corrected_labels], ['ground truth', 'New'])

    def object_evaluation(self, reference_path=None, prediction_path=None):
        self.set_array("cytomask")
        self.set_array("deep_mask")
        self.set_array("deep_labels")
        self.set_array("clean_deep_labels")
        self.set_array("sphere_labels")
        self.set_array("convex_labels")

        # corrected_labels = self.sphere_labels.copy()
        # corrected_labels = self.convex_labels.copy()
        # corrected_labels = self.deep_mask.copy()
        corrected_labels = self.convex_labels.copy()
        # self.set_array("sphere_labels")
        # if input_array_name == 'last_output_array_name':
        #     input_array_name = self.last_output_array_name
        # self.set_array(input_array_name)
        # input_array_name = 'last_output_array_name'
        if reference_path == None:
            reference_path = mrc_cleaner.ask_file_path(self.dir, file_extension=('.mrc'))
            print(reference_path)
        else:
            reference_path = self.dir / reference_path
            print(reference_path)

        # if prediction_path == None:
        #     prediction_path = mrc_cleaner.ask_file_path(self.save_dir, file_extension=('.mrc'))
        #     prediction = mrcfile.open(prediction_path)
        # else:
        #     prediction_path = mrc_cleaner.ask_file_path(self.dir / prediction_path, file_extension=('.mrc'))
        #     prediction = mrcfile.open(prediction_path)
        # prediction = self.last_output_array

        reference = mrcfile.open(reference_path)
        # maskfile = mrcfile.open('./compare/labels_manual.mrc')
        # corrected = mrcfile.open('./compare/labels_automation.mrc')

        # real_image = umic.load_raw('./compare/Dummy_80.rec.nad')
        real_image = self.image
        reff = reference.data
        reff = reff * self.cytomask
        print(np.shape(reff))
        reff = reff >= 10

        # corrected_labels = prediction.data
        # corrected_labels = skimage.morphology.label(corrected_labels, connectivity=1)
        evaluator1 = evaluation_class.ConfusionMatrix(self.deep_mask, reff)
        evaluator2 = evaluation_class.ConfusionMatrix(self.deep_labels >= 1, reff)
        evaluator3 = evaluation_class.ConfusionMatrix(self.clean_deep_labels >= 1, reff)
        evaluator4 = evaluation_class.ConfusionMatrix(self.sphere_labels >= 1, reff)
        evaluator5 = evaluation_class.ConfusionMatrix(self.convex_labels >= 1, reff)
        # print(evaluator)

        print(evaluator1.former_dice())
        print(evaluator2.former_dice())
        print(evaluator3.former_dice())
        print(evaluator4.former_dice())
        print(evaluator5.former_dice())

        reff = skimage.morphology.label(reff, connectivity=1)

        a = []
        a0 = [evaluator1.former_dice(), evaluator2.former_dice(), evaluator3.former_dice(), evaluator4.former_dice(), evaluator5.former_dice()]
        a += a0
        # for ppp in [0.0]:
        #     a1 = cryovesnet.objectwise_evalution(reff, corrected_labels, proportion=ppp)
        #     a2 = cryovesnet.objectwise_evalution(corrected_labels, reff, proportion=ppp)
        #     a += a1
        #     a += a2
        a += cryovesnet.new_objectwise_evalution(reff, corrected_labels)
        # for prediction in [self.deep_labels, self.clean_deep_labels, self.sphere_labels, self.convex_labels]:
        #     a += cryovesnet.new_objectwise_evalution(reff, prediction)

        # #binarize reff and corrected_labels
        # reff = (reff >= 1).astype(int)
        # # corrected_labels = (corrected_labels >= 1).astype(int)
        # true_labels = reff.flatten()
        #
        # predicted_probs = self.deep_mask.copy().flatten()
        # #
        # # Calculate ROC curve
        # fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        #
        # # Calculate AUC
        # roc_auc = auc(fpr, tpr)
        # # Plot ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # # plt.savefig('roc_curve.png')
        # plt.show()

        self.clear_memory()
        # return a,true_labels,predicted_probs
        return a