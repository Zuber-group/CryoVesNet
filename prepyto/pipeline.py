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
import prepyto
from . import pyto_scripts
from .unetmic.unetmic import segment as segseg
from . import weights
#import evaluation_class
from . import mrc_cleaner
from . import visualization
import numpy as np
import pandas as pd
import mrcfile
from tqdm import tqdm
from collections.abc import Iterable

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ModuleNotFoundError:
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("pipeline: tensorflow is not installed, some function will not work.")


if platform.system() == 'Darwin': os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Pipeline():

    def __init__(self,path_to_folder):
        #BZ replaced self.path_to_folder by self.dir
        self.dir = Path(path_to_folder).absolute()
        print(f"Prepyto Pipeline: the pipeline is created for {self.dir}")
        #BZ to get a clearer overview of data structure, all directory paths are defined here
        self.deep_dir = self.dir/'deep'
        self.save_dir = self.dir/'prepyto'
        self.pyto_dir = self.dir/'pyto'
        pattern = '*.rec.nad'
        try:
            self.image_path = next(self.dir.glob(pattern)) #was earlier called path_to_file
        except StopIteration:
            print(f"Error: {self.dir} does not contain file matching the pattern {pattern}. Exiting")
            sys.exit(1)
        #self.binned_deep_mask_path = self.deep_dir/(self.image_path.stem + '_wreal_mask.tiff')
        #there are issues loading a 3D 64-bit tiff --> shape comes incorrect. Therefore we use the np array save on disk
        self.binned_deep_mask_path = self.deep_dir / (self.image_path.stem + '_segUnet.npy')
        self.deep_mask_path = self.save_dir/(self.image_path.stem + '_zoomed_mask.mrc')
        self.cytomask_path = self.save_dir/(self.image_path.stem + '_cytomask.mrc')
        self.active_zone_mask_path = self.save_dir/(self.image_path.stem + '_azmask.mrc')
        self.deep_labels_path = self.save_dir/(self.image_path.stem + '_deep_labels.mrc')
        self.threshold_tuned_labels_path = self.save_dir/(self.image_path.stem + '_intensity_corrected_labels.mrc')
        self.convex_labels_path = self.save_dir/(self.image_path.stem + '_convex_labels.mrc')
        self.mancorr_labels_path = self.save_dir/(self.image_path.stem + '_mancorr.mrc')
        self.sphere_labels_path = self.save_dir/(self.image_path.stem + '_sphere.mrc')
        self.sphere_df_path = self.save_dir/(self.image_path.stem + '_sphere_dataframe.pkl')
        self.no_small_labels_path = self.save_dir/(self.image_path.stem + '_no_small_labels.mrc')
        self.final_vesicle_labels_path = self.save_dir/(self.image_path.stem + '_final_vesicle_labels.mrc')
        self.full_labels_path = self.save_dir/'labels.mrc'

        self.image = None #placeholder for real image array
        self.binned_deep_mask = None #placeholder for binned_deep_mask
        self.deep_mask = None #placeholder for unet mask image array
        self.cytomask = None
        self.active_zone_mask = None
        self.deep_labels = None #placeholder for cleaned label array
        self.threshold_tuned_labels = None #placeholder for threshold tuned label array
        self.convex_labels = None
        self.mancorr_labels = None
        self.sphere_labels = None
        self.no_small_labels = None
        self.final_vesicle_labels = None
        self.full_labels = None
        self.garbage_collector = {'image', 'binned_deep_mask', 'deep_mask', 'cytomask',
                                  'active_zone_mask', 'deep_labels',
                                  'threshold_tuned_labels', 'convex_labels', 'mancorr_labels', 'sphere_labels',
                                  'no_small_labels', 'final_vesicle_labels'}
        self.vesicle_mod_path = self.dir/'vesicles.mod'
        self.active_zone_mod_path = self.dir/'active_zone.mod'
        self.cell_outline_mod_path = self.dir/'cell_outline.mod'
        self.full_mod_path = self.save_dir/'full_prepyto.mod'
        with pkg_resources.path(weights, 'weights.h5') as p:
            self.unet_weight_path = p
        self.check_files()
        self.image_shape = mrcfile.utils.data_shape_from_header(mrcfile.open(self.image_path, header_only=True).header)
        self.voxel_size = prepyto.get_voxel_size_in_nm(self.image_path)
        self.min_radius = prepyto.min_radius_of_vesicle(self.voxel_size)
        self.min_vol = prepyto.min_volume_of_vesicle(self.voxel_size)

        if self.image_path.exists():
            self.set_array('image')
        if self.sphere_df_path.exists():
            self.sphere_df = pd.read_pickle(self.sphere_df_path)

    def quick_setup(self, labels_to_load=['sphere_labels']):
        self.set_array('image')
        for label_name in labels_to_load:
            self.set_array(label_name)

    def set_array(self, array_name, datatype = None):
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
            #warnings.warn("Pipeline.set_array. When changing datatype, data is currently not rescaled, potentially leading to unexpected results.")
            setattr(self, array_name, getattr(self, array_name).astype(datatype))
        return array_name

    def set_segmentation_region_from_mod(self, datatype = np.uint8, force_generate = False):
        if (not self.cytomask_path.exists()) or force_generate:
            cmd = f"imodmop -mode 6 -o 1 -mask 1 \"{self.cell_outline_mod_path}\" \"{self.image_path}\" \"{self.cytomask_path}\""
            os.system(cmd)
        self.set_array('cytomask', datatype=datatype)

    def set_active_zone_array_from_mod(self, datatype = np.uint8, force_generate = False):
        if (not self.active_zone_mask_path.exists()) or force_generate:
            cmd = f"imodmop -mode 6 -o 1 -tube 1 -diam 3 -pl -mask 1 \"{self.active_zone_mod_path}\" \"{self.image_path}\" \"{self.active_zone_mask_path}\""
            os.system(cmd)
        self.set_array('active_zone_mask', datatype=datatype)

    def get_additional_non_vesicle_objects_from_mod(self, mod_objects, destination_labels, modfile_path = None,
                                                    datatype = np.uint16):
        if not modfile_path: modfile_path = self.full_mod_path
        temp_label_path = self.save_dir/'temp_label.mrc'
        mod_objects = self.to_list(mod_objects)
        destination_labels = self.to_list(destination_labels)
        cmd = f"imodmop -mode 0 -objects {','.join(str(e) for e in mod_objects)} -labels {','.join(str(e) for e in destination_labels)}  \"{modfile_path}\" \"{self.image_path}\" \"{temp_label_path}\""
        os.system(cmd)
        with mrcfile.open(temp_label_path) as mrc:
            label_array = mrc.data.astype(datatype)
        temp_label_path.unlink()
        return label_array

    def get_vesicles_from_mod(self, mod_objects, modfile_path = None, datatype = np.uint16):
        if not modfile_path: modfile_path = self.full_mod_path
        temp_label_path = self.save_dir/'temp_label.mrc'
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

    def check_files(self, file_list=['dir', 'image_path', 'active_zone_mod_path',
                                     'cell_outline_mod_path']):
        """
        Check if all input files as specified in file_list exist
        """
        missing = []
        for p_str in file_list:
            p = getattr(self,p_str)
            if not p.exists(): missing.append(str(p))
        if len(missing):
            err = f"""the following file(s) and/or directory(s) are missing: {", ".join(missing)}"""
            raise IOError(err)

    def run_deep(self, force_run=False, rescale=0.5):
        """
        Merged vesicle_segmentation and run_deep to make it a pipeline method
        all output files are saved in self.deep_dir
        """
        print("Prepyto pipeline: Running unet segmentation if there are less than 7 file in ./deep directory")
        #if not os.path.exists('deep') or len(list(os.walk('./deep'))[0][2])<7:
        if self.deep_dir.exists() and len(list(self.deep_dir.glob('*'))) >= 7 and not force_run:
            return
        #create deep_dir (if it does not exist)
        self.deep_dir.mkdir(exist_ok=True)
        #delete every file in deep_dir (not that if it contains directory it will raise a PermissionError)
        for p in self.deep_dir.glob("*"):
            p.unlink()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        tf.keras.backend.set_session(tf.Session(config=config))
        network_size = 64
        segseg.full_segmentation(network_size, str(self.unet_weight_path.absolute()), self.image_path,
                                  self.deep_dir, rescale=rescale, gauss=True)

    def setup_prepyto_dir(self, make_masks = True, memkill = True):
        print("Prepyto Pipeline: setting up prepyto directory")
        self.save_dir.mkdir(exist_ok=True)
        if make_masks:
            self.set_segmentation_region_from_mod()
            self.set_active_zone_array_from_mod()
        if memkill:
            self.clear_memory()

    def zoom(self, force_run=False, memkill=True):
        """
        Zoom the deep mask
        :param memkill:
        :return:
        """
        print("Prepyto Pipeline: zooming the unet mask")
        self.last_output_array_name = 'deep_mask'
        if self.deep_mask_path.exists() and not force_run:
            print("Skipping because a full sized deep mask is already saved on the disk.")
            print("If you want to force the program to make a new full sized deep mask, set force_run to True.")
            return
        self.binned_deep_mask = np.load(self.binned_deep_mask_path)
        self.set_array('image')
        self.deep_mask = skimage.transform.resize(self.binned_deep_mask, output_shape=np.shape(self.image),
                                                  preserve_range=True).astype(np.float32)

        prepyto.save_label_to_mrc(self.deep_mask, self.deep_mask_path, template_path=self.image_path)
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def label_vesicles(self, input_array_name='deep_mask', within_segmentation_region = True,
                       memkill=True):
        """label vesicles from the zoomed deep mask
        :param input_array_name: name of the array to be labelled
        :param within_bound: restrict labelling to segmentation_region? (currently called cytomask)
        this is done by setting every voxel outside the bounding volume to 0 in deep_mask ("_zoomed_mask.mrc)
        :param memkill:
        :return:
        """
        print("Prepyto Pipeline: running label_vesicles")
        self.set_array('image')
        self.set_array(input_array_name)
        self.deep_mask = getattr(self, input_array_name)
        if within_segmentation_region:
            self.outcell_remover(input_array_name='deep_mask', output_array_name='deep_mask', memkill=False)
        opt_th, _ = segseg.find_threshold(self.image, self.deep_mask)
        image_label_opt = skimage.morphology.label(self.deep_mask > opt_th)
        _, deep_labels = segseg.mask_clean_up(image_label_opt)
        self.deep_labels = deep_labels.astype(np.uint16)
        prepyto.save_label_to_mrc(self.deep_labels, self.deep_labels_path, template_path=self.image_path)
        #free up memory
        self.last_output_array_name = 'deep_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def outcell_remover(self, input_array_name='last_output_array_name',
                        output_array_name='deep_labels', memkill=True, force_generate = False):
        print("Prepyto Pipeline: restricting labels to segmentation region")
        self.set_segmentation_region_from_mod()
        self.set_array(input_array_name)
        bounded_array = getattr(self, input_array_name) * self.cytomask
        setattr(self, output_array_name, bounded_array)
        output_path_name = output_array_name + '_path'
        prepyto.save_label_to_mrc(bounded_array, getattr(self,output_path_name), template_path=self.image_path)
        #free up memory
        if memkill:
            self.clear_memory(exclude=[output_array_name, 'image'])
        self.last_output_array_name = output_array_name
        self.print_output_info()

    def threshold_tuner(self, input_array_name='last_output_array_name', memkill=True):
        print('Prepyto Pipeline: running threshold_tuner')
        self.set_array('image')
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.set_array('deep_mask')
        self.threshold_tuned_labels, mean_shell_arg = prepyto.find_threshold_per_vesicle_intensity(int_image=self.image,
                                                                                                   image_label=getattr(self, input_array_name),
                                                                                                   mask_image=self.deep_mask, dilation=0,
                                                                                                   convex=0,
                                                                                                   minimum_volume_of_vesicle=self.min_vol)
        self.threshold_tuned_labels = prepyto.oneToOneCorrection(getattr(self, input_array_name), self.threshold_tuned_labels)
        prepyto.save_label_to_mrc(self.threshold_tuned_labels, self.threshold_tuned_labels_path,
                                  template_path=self.image_path)
        #free up memory
        self.last_output_array_name = 'threshold_tuned_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def label_convexer(self, input_array_name='last_output_array_name', memkill=True):
        print("Prepyto Pipeline: making vesicles convex")
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.convex_labels = prepyto.fast_pacman_killer(getattr(self, input_array_name))
        prepyto.save_label_to_mrc(self.convex_labels, self.convex_labels_path, template_path=self.image_path)
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
        print("Prepyto Pipeline: interactive cleaning. To continue close the napari window.")
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.mancorr_labels = mrc_cleaner.interactive_cleaner(self.image, getattr(self, input_array_name))
        prepyto.save_label_to_mrc(self.mancorr_labels, self.mancorr_labels_path, template_path=self.image_path)
        #free up memory
        self.last_output_array_name = 'mancorr_labels'
        if memkill:
            self.clear_memory(exclude=self.last_output_array_name)
        self.print_output_info()

    def compute_sphere_dataframe(self, input_array_name='last_output_array_name'):
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        sphere_df = prepyto.get_sphere_dataframe(self.image, getattr(self, input_array_name))
        mahalanobis_series = prepyto.mahalanobis_distances(sphere_df.drop(['center'], axis=1))
        sphere_df['mahalanobis'] = mahalanobis_series
        sphere_df.to_pickle(self.sphere_df_path)
        self.sphere_df = sphere_df

    def make_spheres(self, input_array_name='last_output_array_name', memkill=True):
        print("Prepyto Pipeline: Making vesicles spherical.")
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        self.compute_sphere_dataframe(input_array_name)
        self.sphere_labels = prepyto.make_vesicle_from_sphere_dataframe(getattr(self,input_array_name), self.sphere_df)
        prepyto.save_label_to_mrc(self.sphere_labels, self.sphere_labels_path, template_path=self.image_path)
        self.last_output_array_name = 'sphere_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()

    def identify_spheres_outliers(self, bins=50, min_mahalanobis_distance=2.0):
        ax = self.sphere_df.mahalanobis.hist(bins=bins, color='blue')
        ylim0, ylim1 = ax.get_ylim()
        ax.vlines(min_mahalanobis_distance, ylim0, ylim1, linestyles='dashed', color='red',
                  label = 'Mahalanobis threshold for printing sphere details')
        ax.set_title("Outlier spheres detection")
        ax.set_xlabel("Mahalanobis")
        ax.set_ylabel("Number of spheres")
        print(self.sphere_df[self.sphere_df.mahalanobis > min_mahalanobis_distance].sort_values('mahalanobis'))
        print("You should inspect the labels that have a high mahalanobis distance as they are the likeliest to be wrongly segmented")

    def fix_spheres_interactively(self, input_array_name='last_output_array_name', memkill=True):
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        points_to_remove, points_to_add, points_to_add_sizes = visualization.add_points_remove_labels(self.image,
                                                                                                      getattr(self,input_array_name))
        minimum_box_size = self.voxel_size * 50
        self.mancorr_labels = prepyto.remove_labels_under_points(getattr(self, input_array_name), points_to_remove)
        self.mancorr_labels = prepyto.add_sphere_labels_under_points(self.image,self.mancorr_labels, points_to_add,
                                                                     points_to_add_sizes, minimum_box_size)
        self.last_output_array_name = 'mancorr_labels'
        prepyto.save_label_to_mrc(self.mancorr_labels, self.mancorr_labels_path, template_path=self.image_path)
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
        print("Prepyto Pipeline: removing small labels")
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        label_index, label_size = np.unique(getattr(self, input_array_name), return_counts=True)
        sizes_labels = dict(zip(*(label_index, label_size)))
        small_labels = np.array([k for (k, v) in sizes_labels.items() \
                                 if (v < self.min_vol) and (k >= first_label)])
        selection = np.isin(getattr(self, input_array_name), small_labels)
        self.no_small_labels = getattr(self, input_array_name).copy()
        self.no_small_labels[selection] = 1 #set the small labels to 1, i.e. cytoplasm
        #remove gaps
        missing_labels = [i for i in range(first_label, max(label_index) + 1) if not i in label_index ]
        small_labels = list(set(small_labels).union(missing_labels))
        small_labels.sort()
        for l in tqdm(small_labels[::-1], desc="reordering remaining labels."):
            self.no_small_labels = np.where(self.no_small_labels > l, self.no_small_labels - 1, self.no_small_labels)
        prepyto.save_label_to_mrc(self.no_small_labels, self.no_small_labels_path, template_path=self.image_path)
        #free up memory
        self.last_output_array_name = 'no_small_labels'
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        self.print_output_info()


    def visualization_old_new(self, input_array_name1, input_array_name2, memkill=True):
        print("Prepyto Pipeline: visualizing two sets of labels. To continue, close napari window")
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
        print("Prepyto Pipeline: setting up pyto folder")
        if self.pyto_dir.exists():
            if overwrite:
                self.pyto_dir.unlink()
            else:
                print(f"{self.pyto_dir} exists. Skipping pyto file creation. To force reinitialization of pyto folder, set overwrite to True")
                return
        with pkg_resources.path(pyto_scripts, '.') as src:
            shutil.copytree(src, self.pyto_dir)

    def make_full_modfile(self, input_array_name='last_output_array_name', do_rearrange_labels=True, memkill=True):
        """
        convert the vesicle label file to a modfile and merge it with the initial modfile
        do_rearrange_labels: if True, then no empty label in the input labels array are left
        """
        print("Prepyto Pipeline: making full mod file")
        if input_array_name == 'last_output_array_name':
            input_array_name = self.last_output_array_name
        self.set_array(input_array_name)
        if do_rearrange_labels:
            self.final_vesicle_labels = prepyto.rearrange_labels(getattr(self,input_array_name))
            print("rearranged labels")
        else:
            self.final_vesicle_labels = getattr(self,input_array_name)
        prepyto.save_label_to_mrc(self.final_vesicle_labels, self.final_vesicle_labels_path, template_path=self.image_path)
        self.last_output_array_name = 'final_vesicle_labels'
        self.print_output_info()
        cmd0 = f"imodauto -f 3 -m 1 -h 0 \"{self.final_vesicle_labels_path}\" \"{self.full_mod_path}\""
        cmd1 = f"imodjoin -c \"{self.cell_outline_mod_path}\" \"{self.active_zone_mod_path}\" \"{self.full_mod_path}\" \"{self.full_mod_path}\""
        os.system(cmd0)
        os.system(cmd1)
        if memkill:
            self.clear_memory(exclude=[self.last_output_array_name, 'image'])
        print(f"full model file saved to {self.full_mod_path.relative_to(self.dir)}")


    def handpick(self):
        cmd0 = f"imod -W \"{self.image_path}\" \"{self.full_mod_path}\""
        os.system(cmd0)


    def make_full_label_file(self, vesicle_array_name='last_output_array_name',
                             include_segmentation_region = True, include_active_zone = True,
                             additional_non_vesicle_objects = [], additional_non_vesicle_destination_label = [],
                             additional_non_vesicle_object_modpath = None,
                             handpicked_vesicles_objects = [],
                             hand_picked_vesicles_modpath = None,
                             first_vesicle_index = 10, memkill=True):
        """
        Assemble full label file from masks and full mod file
        :return:
        """
        print("Prepyto Pipeline: making full label file")
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
        vesicle_label_array = skimage.morphology.label(getattr(self,actual_vesicle_array_name))
        vesicle_indices = vesicle_label_array.nonzero()
        offset = first_vesicle_index - 1
        self.full_labels[vesicle_indices] = vesicle_label_array[vesicle_indices] + offset
        if handpicked_vesicles_objects:
            handpicked_array = self.get_vesicles_from_mod(handpicked_vesicles_objects, hand_picked_vesicles_modpath)
            offset = self.full_labels.max()
            handpicked_indices = handpicked_array.nonzero()
            self.full_labels[handpicked_indices] = handpicked_array[handpicked_indices] + offset
        prepyto.save_label_to_mrc(self.full_labels, self.full_labels_path, template_path=self.image_path)
        if memkill:
            self.clear_memory()
        print(f"saved labels to {self.full_labels_path.relative_to(self.dir)}")



    def evaluation(self, reference_path=None, prediction_path=None):
        reference_path=mrc_cleaner.ask_file_path(self.dir,file_extension=('.mrc'))
        if prediction_path==None:
            prediction_path = mrc_cleaner.ask_file_path(self.dir,file_extension=('.mrc'))
            prediction = mrcfile.open(prediction_path)
        else:
            prediction = self.last_output_array

        reference = mrcfile.open(reference_path)
        #maskfile = mrcfile.open('./compare/labels_manual.mrc')
        #corrected = mrcfile.open('./compare/labels_automation.mrc')

        # real_image = umic.load_raw('./compare/Dummy_80.rec.nad')
        real_image= self.image
        mask = reference.data
        print(np.shape(mask))
        mask = mask >= 10
        corrected_labels = prediction.data
        corrected_labels = corrected_labels >= 10
        evaluator = evaluation_class.ConfusionMatrix(corrected_labels, mask)
        print(evaluator)

        print("The Pixel Accuracy is: " + str(evaluator.accuracy()))
        print("The  Intersection-Over-Union is: " + str(evaluator.jaccard_index()))
        print("The Dice Metric: " + str(evaluator.dice_metric()))
        # visualization.viz_labels(real_image, [mask, corrected_labels], ['grand truth', 'New'])