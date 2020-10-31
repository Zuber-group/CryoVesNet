# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys
import skimage
import prepyto
import mrc_cleaner
import visualization
import unetmic.unetmic as umic
import numpy as np
import unetmic.segment as segseg
import mrcfile
from scipy import ndimage
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import complete_vesicle_segmentation




class pipeline():

    def __init__(self,path_to_folder):
        #BZ replaced self.path_to_folder by self.dir
        self.dir = Path(path_to_folder).absolute()
        print(f"PIPELINE: the pipeline is created for {self.dir}")
        #BZ to get a clearer overview of data structure, all directory paths are defined here
        self.deep_dir = self.dir/'deep'
        self.save_dir = self.dir/'prepyto'
        self.cytomask_path = self.dir/'cytomask.mrc'
        pattern = '*.rec.nad'
        try:
            self.image_path = next(self.dir.glob(pattern))
        except StopIteration:
            print(f"Error: {self.dir} does not contain file matching the pattern {pattern}. Exiting")
            sys.exit(1)
        self.mancorr_path = self.save_dir/(self.image_path.stem + '_mancorr.mrc')
        self.mancorr_clean_path = self.save_dir/(self.mancorr_path.stem + '_clean.mrc')
        self.mylabel_path = self.save_dir/(self.image_path.stem + '_overall_ourcorrected_labels.mrc')
        self.mylabel_mancorr_path = self.save_dir/(self.mylabel_path.stem + '_mancorr.mrc')
        self.mylabel_mancorr_clean_path = self.save_dir/(self.mylabel_mancorr_path.stem + '_clean.mrc')
        self.sphere_path = self.save_dir/(self.image_path.stem + '_sphere.mrc')
        self.labels_mrc_path = self.save_dir/'labels.mrc'
        self.last_mrc_path = None #placeholder for more flexibility in pipeline sequence
        self.real_image = None #placeholder for real image array
        self.deep_mask_image = None #placeholder for unet mask image array
        self.corrected_labels = None #placeholder for corrected label array
        self.clean_labels = None #placeholder for cleaned label array
        self.vesicle_mod_path = self.dir/'vesicles.mod'
        self.active_zone_mod_path = self.dir/'active_zone.mod'
        self.cell_outline_mod_path = self.dir/'cell_outline.mod'
        self.full_mod_path = self.dir/'merge.mod'
        self.check_files()
        self.volume_threshold_of_vesicle = prepyto.min_volume_of_vesicle(self.image_path)
    def check_files(self):
        """
        Check if all input files exist
        """
        missing = []
        for p in (self.dir, self.cytomask_path, self.image_path)
            if not p.exists(): missing.append(str(p))
        if len(missing):
            err = f"""the following file(s) and/or directory(s) are missing: {", ".join(missing)}"""
            raise IOError(err)
    def run_deep(self):
        """
        Merged vesicle_segmentation and run_deep to make it a pipeline method
        all output files are saved in self.deep_dir
        """
        print("PIPELINE: if there is less than 7 file in ./deep directory it gonna generate the output of deep network again, we are in " + os.getcwd())
        print(self.dir)
        #if not os.path.exists('deep') or len(list(os.walk('./deep'))[0][2])<7:
        if self.deep_dir.exists() and len(list(self.deep_dir.glob('*'))) >= 7:
            return
        self.unet_weigth_path = self.deep_dir/'weights.h5'
        #create deep_dir (if it does not exist)
        self.deep_dir.mkdir(exist_ok=True)
        #delete every file in deep_dir (not that if it contains directory it will raise a PermissionError)
        for p in self.deep_dir.glob("*"):
            p.unlink()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        tf.keras.backend.set_session(tf.Session(config=config))
        network_size = 64
        #TODO if we uncomment the next line, we need to check paths.
        #segment.full_segmentation(network_size, self.unet_weigth_path, self.image_path,
        #                          self.deep_dir, rescale=0.5, gauss=True)
    def setup_prepyto_dir(self):
        #image_dir has already been set in __init__
        #it is self.dir
        print("PIPELINE: setup prepyto directory, we are in:" +os.getcwd())
        # if not os.path.exists('prepyto'):
        #     os.mkdir('prepyto')
        #changed naming from self.folder_to_save to self.save_dir to consistently use "dir"
        # for directory.
        self.save_dir.mkdir(exist_ok=True)
        ## image_dir = os.path.splitext(os.path.split(path_to_file)[0])[0]
        #image_dir = '.'
        print(self.dir)
        #self.image_name = os.path.splitext(os.path.split(self.path_to_file)[1])[0]
        ## image_name=file_name
        #self.folder_to_save = image_dir + '/prepyto/'
        #print(self.folder_to_save)
        print(self.save_dir)
        #print(os.path.normpath(self.folder_to_save) + '/' + os.path.splitext(os.path.split(self.path_to_file)[1])[0])
        print(self.image_path)
    def zoom(self):
        print("PIPELINE: here we enlarge the mask to work with real size of the image")#, we are in " + os.getcwd())
        #print(os.getcwd())
        #os.chdir('./deep')
        #cwd = Path('.')
        #mask_path = cwd.glob('*_wreal_mask.tiff')
        #selfmask_name = [str(x) for x in mask_path][0]
        try:
            self.deep_mask_path = next(self.deep_dir.glob('*_wreal_mask.tiff'))
        except StopIteration:
            print(f"""Error: no *_wreal_mask.tiff file found in {self.deep_dir}""")
            sys.exit(1)
        self.deep_mask_image = skimage.io.imread(self.deep_mask_path)
        # cleannmask_image = skimage.io.imread(image_dir + '/deep/Dummy_133_clean_mask.tiff')
        #os.chdir('../..')
        #os.chdir(self.path_to_folder)
        ## path_to_file = './data/133_wtko/Dummy_133.rec.nad.rec'
        ## real_image=skimage.io.imread('./Results_Dummy_133/Dummy_133.rec')
        self.real_image = umic.load_raw(self.image_path)
        #TODO evaluate if we really need to keep the image in memory the whole time
        #TODO same question with the deep_mask_image
        ## int_image = skimage.io.imread(image_dir + '/deep/Dummy_133_processed.tiff')
        ## image_label = skimage.io.imread(image_dir + '/deep/Dummy_133_clean_label.tiff')
        print(f"real image shape is {np.shape(self.real_image)}")
        print(f"deep mask image shape is {np.shape(self.deep_mask_image)}")
        #print(os.path.splitext(os.path.split(self.path_to_file)[1])[0])
        self.deep_mask_image = skimage.transform.resize(self.deep_mask_image, output_shape=np.shape(self.real_image), preserve_range=True)
        # new_image= skimage.transform.resize(int_image, output_shape=np.shape(real_image), preserve_range = True).astype(np.int16)
        new_image = self.real_image.astype(np.int16)

        print(f"new image shape is {np.shape(new_image)}")

        # generate zoomed labels
        opt_th, _ = segseg.find_threshold(new_image, self.mask_image)
        image_label_opt = skimage.morphology.label(self.mask_image > opt_th)
        clean_mask, clean_labels = segseg.mask_clean_up(image_label_opt)

        clean_mask = (255 * clean_mask).astype(np.uint8)
        self.clean_labels = clean_labels.astype(np.uint16)
    def outcell_remover(self):
        print("PIPELINE: we gonna remove outside of the cell labels, we are in " + os.getcwd())
        # print(os.getcwd())
        # os.chdir('./' + image_dir)
        # print(os.getcwd())
        os.system(f"imodmop -mode 1 -o 1 -mask 1 cell_outline.mod {self.image_path} {self.cytomask_path}")
        self.cell_outline = mrcfile.open(self.cytomask_path).data.astype(np.uint16)
        #TODO do we need to keep self.cell_outline in memory the whole time?
        # os.chdir('../..')
        #print(os.getcwd())

        # cell_outline=mrcfile.open('./'+folder_to_save+'/cytomask.mrc').data.astype(np.uint16)
        self.clean_labels = self.clean_labels * self.cell_outline
    def thereshold_tunner(self):#BZ has not touched this function

        # TODO: we can treat with small vesicles outside of this method (or replicate as method to have this functionality separately)
        print("PIPELINE: find find threshold for each vesicle, we are in (its turn off!) " + os.getcwd())
        # prepyto.save_label_to_tiff(clean_labels,path_to_file,folder_to_save,suffix='_corrected_labels')
        prepyto.save_label_to_mrc(self.clean_labels, self.path_to_file, self.folder_to_save, suffix='_corrected_labels')

        corrected_labels, mean_shell_arg = prepyto.find_threshold_per_vesicle_intensity(int_image=self.real_image,
                                                                                        image_label=self.clean_labels,
                                                                                        mask_image=self.mask_image, dilation=0,
                                                                                        convex=0,
                                                                                        volume_of_vesicle=self.volume_threshold_of_vesicle)

        # corrected_labels = prepyto.fast_pacman_killer(corrected_labels)


        # prepyto.save_label_to_tiff(corrected_labels, path_to_file, folder_to_save, suffix='_intensity_corrected_labels')
        prepyto.save_label_to_mrc(corrected_labels, self.path_to_file, self.folder_to_save, suffix='_intensity_corrected_labels')
        #
        old_label_path = self.folder_to_save + self.image_name + '_corrected_labels.mrc'
        mylabel_path = self.folder_to_save + self.image_name + '_intensity_corrected_labels.mrc'

        old_labels = mrcfile.open(old_label_path).data.astype(np.uint16)
        myimage_labels = mrcfile.open(mylabel_path).data.astype(np.uint16)

        self.corrected_labels = prepyto.oneToOneCorrection(old_labels, myimage_labels)

        # prepyto.save_label_to_tiff(corrected_labels,path_to_file,folder_to_save,suffix='_overall_ourcorrected_labels')
        prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, self.folder_to_save, suffix='_overall_ourcorrected_labels')
    def label_convexer(self):
        print("PIPELINE: pacman killer!, we are in (its turn off!)  " + os.getcwd())
        #mylabel_path = self.folder_to_save + os.path.splitext(os.path.split(self.path_to_file)[1])[
        #    0] + '_overall_ourcorrected_labels.mrc'
        #BZ moved path definition to __init__
        self.myimage_labels = mrcfile.open(self.mylabel_path).data.astype(np.uint16)
        self.corrected_labels = self.myimage_labels
        # self.corrected_labels = prepyto.fast_pacman_killer(self.myimage_labels)

        # prepyto.save_label_to_tiff(corrected_labels, path_to_file, folder_to_save, suffix='_intensity_corrected_labels')
        #prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, self.folder_to_save, suffix='_overall_ourcorrected_labels')
        prepyto.save_label_to_mrc(self.corrected_labels, self.mylabel_path)
    def interactive_cleaner(self, corrected_labels_as_input=True):
        """
        starts interactive cleaner
        :param corrected_labels_as_input: if True, use self.corrected_labels as input, else use self.myimage_labels
        :return:
        """
        print("PIPELINE: interactive cleaning till close the nappari, we are in " + os.getcwd())

        # # image_path = image_path = 'mask_133/Dummy_133_processed.tiff'
        #mylabel_path = self.folder_to_save + os.path.splitext(os.path.split(self.path_to_file)[1])[
        #    0] + '_overall_ourcorrected_labels.mrc'
        #myimage_labels = mrcfile.open(mylabel_path).data.astype(np.uint16)
        #target_labels = myimage_labels
        if corrected_labels_as_input:
            input_labels = self.corrected_labels
            output_path = self.mylabel_mancorr_path
            output_clean_path = self.mylabel_mancorr_clean_path
        else:
            input_labels = self.myimage_labels
            output_path = self.mancorr_path
            output_clean_path = self.mancorr_clean_path
        new_labels = mrc_cleaner.interactive_cleaner(self.real_image, input_labels)
        prepyto.save_label_to_mrc(new_labels, output_path)
        mrc_cleaner.mrc_header_cleaner(output_path, self.image_path, output_clean_path)
    def sphere_vesicles(self):
        print("PIPELINE: this method fit sphere on vesicles, we are in " + os.getcwd())
        target_labels = mrcfile.open(self.mylabel_mancorr_clean_path).data.astype(np.uint16)
        self.corrected_labels = prepyto.ellipsoid_vesicles(image_label=target_labels, diOrEr=0)
        #TODO do we need to keep self.corrected_labels in memory?
        #prepyto.save_label_to_tiff(clean_labels,path_to_file,folder_to_save,suffix='_ellipsoid_corrected_labels')
        #path_to_last_mrc = prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, self.folder_to_save,
        #                                             suffix='_ellipsoid_corrected_labels')
        #path_to_last_mrc = prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, '.',
        #                                             suffix='_ellipsoid_corrected_labels')
        prepyto.save_label_to_mrc(self.corrected_labels, self.sphere_path)
        self.last_mrc_path = mrc_cleaner.mrc_header_cleaner(self.sphere_path, self.image_path, self.sphere_path)
    def visualization_old_new(self):
        # TODO: this method is better to get some argument instead of static argument and show them in nappari
        print("PIPELINE: visualization , we are in " + os.getcwd())
        visualization.viz_labels(self.real_image, [self.clean_labels, self.corrected_labels], ['Old', 'New'])
    def making_pyto_stuff(self):
        # TODO: we should separate this method to smaller method for reproducibilty!
        print("PIPELINE: generate pyto files , we are in " + os.getcwd())
        #TODO: What is the point of the next line?
        mrc_cleaner.query_yes_no("continue?")
        # os.chdir('./')
        print(os.getcwd())
        #TODO: What is the point of the next line?
        yesOrNo = mrc_cleaner.query_yes_no("We can wait here for a while! do you want to pick by hand also?")
        print(f"imodauto -f 3 -m 1 -h 0 {self.last_mrc_path} {self.vesicle_mod_path}")
        os.system(f"imodauto -f 3 -m 1 -h 0 {self.last_mrc_path} {self.vesicles_mod_path}")
        os.system(f"imodjoin -c {self.cell_outline_mod_path} {self.active_zone_mod_path} merge1.mod")
        os.system(f"imodjoin -c merge1.mod {self.vesicle_mod_path} merge2.mod")
        handPick = False
        while yesOrNo:
            # these thing not general for all the files!!!!!
            os.system(f'imod -W {self.img_path} merge2.mod')
            handPick = mrc_cleaner.query_yes_no("Is the mod file ready?")
            if handPick:
                break
            else:
                yesOrNo = mrc_cleaner.query_yes_no("We can wait here for a while! do you want to pick by hand also?")
        # os.chdir('../..')
        print(os.getcwd())
       # sv = 'Results_Vesicles/20200615/bubu_ellipsoid.mod'

        # sv_mask = './before_sphere.mrc'
        sv_mask = self.last_mrc_path

        # sv_mask_zoom_out = './Dummy_133_ellipsoid_corrected_labels.mrc'
        # sv_mask_CV = 'mancorr_sv.mrc'



        base_dir = os.path.abspath('.')
        pyto_base_dir = os.path.join(base_dir, '')
        tomo_dir = os.path.join(pyto_base_dir, '3d')
        common_dir = os.path.join(pyto_base_dir, 'common')
        connectors_dir = os.path.join(pyto_base_dir, 'connectors')
        layers_dir = os.path.join(pyto_base_dir, 'layers')
        vesicles_dir = os.path.join(pyto_base_dir, 'vesicles')

        sv = mrcfile.open(sv_mask)
        sv_label, nlabel_DV = ndimage.label((sv.data))
        sv_label = np.where(sv_label > 0, sv_label + 9, 0)  ### changed 0 to 1 in last place
        sizes_sv = dict(zip(*np.unique(sv_label, return_counts=True)))

        # os.chdir('./' + image_dir)
        print(os.getcwd())
        os.system("imodmop -mode 1 -o 1 -tube 1 -diam 3 -pl -mask 1 active_zone.mod "+self.file_name+" azmask.mrc")

        cytosol = mrcfile.open(cytomask)
        labels = cytosol.data
        activezone = mrcfile.open(az_mask)
        initial_labels = np.maximum(labels, activezone.data * 2)
        # os.chdir('../..')
        # print(os.getcwd())

        if handPick:
            # os.chdir('./' + image_dir)
            print("which objects contain the handpicked vesicles?if you enter nothing it will assume object number 4)")
            print("write numbers with comma like: 5,7")
            choice = input()
            choice_tupple = eval(choice)
            if len(choice_tupple) == 0:
                choice_tupple = (4,)
            choice_tupple = (0,) + choice_tupple

            print(os.getcwd())
            os.system(
                "imodmop -mode 1 -o " + str(choice_tupple)[
                                        4:-1] + " -mask 1 -3 merge2.mod " + self.file_name + " mancorr_sv.mrc")  # diffrent object handeling 4,5,6
            sv_mask_CV = 'mancorr_sv.mrc'



            print(os.getcwd())
            os.system(
                "imodmop -mode 1 -o 4 -mask 1 -3 merge2.mod "+self.file_name+" mancorr_sv.mrc")  # diffrent object handeling 4,5,6
            sv_mask_CV = 'mancorr_sv.mrc'

            sv_CV = mrcfile.open(sv_mask_CV)
            sv_CV_label, nlabel_CV = ndimage.label(sv_CV.data)
            sv_CV_label = np.where(sv_CV_label > 0, sv_CV_label + sv_label.max(), 0)
            sv_label = np.maximum(sv_CV_label, sv_label)
            labels = np.maximum(initial_labels, sv_label)
            labels = labels.astype(np.int16)
            # os.chdir('../..')
            # print(os.getcwd())

        sv_mask = sv_label.astype(np.int16)
        labels = np.maximum(initial_labels, sv_mask)
        print(f"number of vesicles: {sv_label.max() - 9}")

        with mrcfile.new(self.labels_mrc_path, overwrite=True) as mrc:
            tomo = mrcfile.open(self.last_mrc_path)
            mrc.set_data(labels)
            mrc.voxel_size = tomo.voxel_size

        # double check
        labels = mrcfile.open(self.labels_mrc_path).data
        label_index, label_size = np.unique(labels, return_counts=True)
        sizes_labels = dict(zip(*(label_index, label_size)))

        for i in range(max(label_index) + 1):
            if not i in label_index:
                print(i)

        min_vol=prepyto.min_volume_of_vesicle(self.path_to_file)
        small_labels = np.array([k for (k, v) in sizes_labels.items() if v < min_vol])
        # print(small_labels)
        selection = np.isin(labels, small_labels)
        temp_labels = labels.copy()
        temp_labels[selection] = 1

        # which indices are missing?
        missing_sv = []
        for i in range(10, max(label_index) + 1):
            if not i in label_index:
                missing_sv.append(i)

        small_labels = list(set(small_labels).union(missing_sv))
        small_labels.sort()

        for l in tqdm(small_labels[::-1]):
            temp_labels = np.where(temp_labels > l, temp_labels - 1, temp_labels)

        # create new lists and dictionary where the small labels have been removed
        temp_label_index, temp_label_size = np.unique(temp_labels, return_counts=True)
        temp_sizes_labels = dict(zip(*(temp_label_index, temp_label_size)))
        print(f"number of vesicles: {max(temp_sizes_labels.keys()) - 9}")

        sv_sizes_labels = temp_sizes_labels.copy()
        for k in range(3):
            try:
                del sv_sizes_labels[k]
            except:
                pass
        sv_sizes_labels
        # sort keys according to value (i.e. sort original index numbers according to label size)
        sorted_sv_sizes_labels = {k: v for k, v in sorted(sv_sizes_labels.items(), key=lambda item: item[1])}

        # make the lookup table (we do not change labels 0 to 9)
        lookup = list(range(10)) + list(sorted_sv_sizes_labels.keys())
        lookup = np.array(lookup)

        myLookup = np.arange(len(lookup))
        myLookup[lookup] = np.array(range(len(lookup)))

        new_labels = myLookup[temp_labels]
        # in future save to labels_out but for now we use another filename
        # and open tomogram instead of 3d/Dummy_133.mrc:
        with mrcfile.new(self.labels_mrc_path, overwrite=True) as mrc:
            tomo = mrcfile.open(self.last_mrc_path)
            mrc.set_data(new_labels.astype(np.int16, copy=False))
            mrc.voxel_size = tomo.voxel_size
            mrc.header['origin'] = tomo.header['origin']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_dir = Path().absolute()
    try:
        os.chdir('data')
    except FileNotFoundError:
        print(f"Error: {base_dir} does not contain a \"data\" sub-directory called. Exiting")
        sys.exit(1)
    #dataset_dir = os.getcwd()
    dataset_dir = Path().absolute()
    #datasetname_lst=os.listdir('.')
    #dataset_lst=[dataset_dir+x for x in datasetname_lst]
    dataset_lst = [e for e in dataset_dir.iterdir()]
    while True:
        print(f"Which dataset are we working on? Please enter a number between 0 and {len(dataset_lst)-1}")
        #[print(x) for x.name in dataset_lst]
        for i, p in enumerate(dataset_lst):
            print(f"{i}: {p.name}")
        print("?")
        try:
            choice = int(input())
            dataset_lst[choice]
            break #if no exception has been triggered, then exit the while loop and execute the rest
        except (ValueError, KeyError):
            print(f"Error: please chose a number between 0 and and {len(dataset_lst)-1}")
    #working_datset_index=datasetname_lst.index(choice)
    # print(working_datset_index)
    #myPipeline=pipeline(datasetname_lst[working_datset_index])
    myPipeline = pipeline(dataset_lst[choice])
    # for dataset in dataset_lst:
    #     pipeline(dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    myPipeline.run_deep() #BZ cleaned up (except segment.full_segmentation)
    myPipeline.setup_prepyto_dir() #BZ cleaned up
    myPipeline.zoom() #BZ cleaned up (see if some of the object properties (np array))
    myPipeline.outcell_remover() #BZ cleaned up
    # myPipeline.thereshold_tunner()
    myPipeline.label_convexer() #BZ cleaned up (not it does not run pacman killer)
    myPipeline.interactive_cleaner() #BZ cleaned up
    myPipeline.sphere_vesicles() #BZ cleaned up
    myPipeline.visualization_old_new() #BZ cleaned up
    myPipeline.making_pyto_stuff() #BZ cleaned up




















