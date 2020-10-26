# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
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

        print("PIPELINE: the pipeline is created for ",path_to_folder)
        print(os.getcwd())
        self.path_to_folder=path_to_folder

        os.chdir(self.path_to_folder)
        cwd = Path('.')
        path_generator = cwd.glob('*.rec.nad')
        file_name = ([str(x) for x in path_generator])[0]
        # file_name = [x.stem for x in path_generator]
        print(file_name)
        self.file_name=file_name
        self.path_to_file = './' + file_name
        print(self.path_to_file)
        self.volume_threshold_of_vesicle = prepyto.min_volume_of_vesicle(self.path_to_file)
        print(os.getcwd())

    def run_deep(self):
        print("PIPELINE: if there is less than 7 file in ./deep directory it gonna generate the oytput of deep network again, we are in " + os.getcwd())
        print(self.path_to_folder)
        if not os.path.exists('deep') or len(list(os.walk('./deep'))[0][2])<7:
            complete_vesicle_segmentation.vesicle_segmentation(self.path_to_folder)

    def setup_prepyto_dir(self):
        print("PIPELINE: setup prepyto directory, we are in:" +os.getcwd())
        if not os.path.exists('prepyto'):
            os.mkdir('prepyto')

        # image_dir = os.path.splitext(os.path.split(path_to_file)[0])[0]
        image_dir = '.'
        print(image_dir)
        self.image_name = os.path.splitext(os.path.split(self.path_to_file)[1])[0]
        # image_name=file_name
        self.folder_to_save = image_dir + '/prepyto/'
        print(self.folder_to_save)
        print(os.path.normpath(self.folder_to_save) + '/' + os.path.splitext(os.path.split(self.path_to_file)[1])[0])

    def zoom(self):
        print("PIPELINE: here we enlarge the mask to work with real size of the image, we are in " + os.getcwd())
        print(os.getcwd())
        os.chdir('./deep')
        cwd = Path('.')
        mask_path = cwd.glob('*_wreal_mask.tiff')
        mask_name = [str(x) for x in mask_path][0]

        self.mask_image = skimage.io.imread(mask_name)
        # cleannmask_image = skimage.io.imread(image_dir + '/deep/Dummy_133_clean_mask.tiff')
        os.chdir('../..')
        os.chdir(self.path_to_folder)

        # path_to_file = './data/133_wtko/Dummy_133.rec.nad.rec'




        # real_image=skimage.io.imread('./Results_Dummy_133/Dummy_133.rec')

        self.real_image = umic.load_raw(self.path_to_file)
        # int_image = skimage.io.imread(image_dir + '/deep/Dummy_133_processed.tiff')
        # image_label = skimage.io.imread(image_dir + '/deep/Dummy_133_clean_label.tiff')

        print(np.shape(self.real_image))
        print(np.shape(self.mask_image))
        print(os.path.splitext(os.path.split(self.path_to_file)[1])[0])

        self.mask_image = skimage.transform.resize(self.mask_image, output_shape=np.shape(self.real_image), preserve_range=True)
        # new_image= skimage.transform.resize(int_image, output_shape=np.shape(real_image), preserve_range = True).astype(np.int16)
        new_image = self.real_image.astype(np.int16)

        print(np.shape(new_image))

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
        os.system('imodmop -mode 1 -o 1 -mask 1 cell_outline.mod '+self.image_name+' cytomask.mrc')
        self.cell_outline = mrcfile.open('./cytomask.mrc').data.astype(np.uint16)
        # os.chdir('../..')
        print(os.getcwd())

        # cell_outline=mrcfile.open('./'+folder_to_save+'/cytomask.mrc').data.astype(np.uint16)
        new_clean_labels = self.clean_labels * self.cell_outline
        self.clean_labels = new_clean_labels



    def thereshold_tunner(self):

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
        mylabel_path = self.folder_to_save + os.path.splitext(os.path.split(self.path_to_file)[1])[
            0] + '_overall_ourcorrected_labels.mrc'
        myimage_labels = mrcfile.open(mylabel_path).data.astype(np.uint16)
        self.corrected_labels=myimage_labels
        # self.corrected_labels = prepyto.fast_pacman_killer(myimage_labels)

        # prepyto.save_label_to_tiff(corrected_labels, path_to_file, folder_to_save, suffix='_intensity_corrected_labels')
        prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, self.folder_to_save, suffix='_overall_ourcorrected_labels')



    def interactive_cleaner(self):
        print("PIPELINE: interactive cleaning till close the nappari, we are in " + os.getcwd())

        # # image_path = image_path = 'mask_133/Dummy_133_processed.tiff'
        mylabel_path = self.folder_to_save + os.path.splitext(os.path.split(self.path_to_file)[1])[
            0] + '_overall_ourcorrected_labels.mrc'
        myimage_labels = mrcfile.open(mylabel_path).data.astype(np.uint16)
        #target_labels = myimage_labels
        new_labels=mrc_cleaner.interactive_cleaner(self.real_image,myimage_labels)

        mylabel_path=prepyto.save_label_to_mrc(new_labels,mylabel_path,self.folder_to_save,'_mancorr')

        mrc_cleaner.mrc_header_cleaner(mylabel_path,self.path_to_file)


    def sphere_vesicles(self):
        print("PIPELINE: this method fit sphere on vesicles, we are in " + os.getcwd())

        target_label_path = self.folder_to_save+ os.path.splitext(os.path.split(self.path_to_file)[1])[0]+'_overall_ourcorrected_labels_mancorr.mrc'

        target_label_path= self.folder_to_save+ os.path.splitext(os.path.split(target_label_path)[1])[0]+'_clean.mrc'
        target_labels = mrcfile.open(target_label_path).data.astype(np.uint16)

        self.corrected_labels = prepyto.elipsoid_vesicles(image_label=target_labels, diOrEr=0)

        # prepyto.save_label_to_tiff(clean_labels,path_to_file,folder_to_save,suffix='_elipsoid_corrected_labels')
        path_to_last_mrc = prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, self.folder_to_save,
                                                     suffix='_elipsoid_corrected_labels')
        path_to_last_mrc = prepyto.save_label_to_mrc(self.corrected_labels, self.path_to_file, '.',
                                                     suffix='_elipsoid_corrected_labels')

        self.last_mrc = mrc_cleaner.mrc_header_cleaner(path_to_last_mrc, self.path_to_file)


    def visualization_old_new(self):
        # TODO: this method is better to get some argument instead of static argument and show them in nappari
        print("PIPELINE: visualization , we are in " + os.getcwd())
        visualization.viz_labels(self.real_image, [self.clean_labels, self.corrected_labels], ['Old', 'New'])


    def making_pyto_stuff(self):
        # TODO: we should seperate this method to smaller method for repreducibilty!
        print("PIPELINE: generate pyto files , we are in " + os.getcwd())
        mrc_cleaner.query_yes_no("continue?")
        # os.chdir('./')
        print(os.getcwd())
        yesOrNo = mrc_cleaner.query_yes_no("We can wait here for a while! do you want to pick by hand also?")
        print("imodauto -f 3 -m 1 -h 0 " + os.path.splitext(os.path.split(self.last_mrc)[1])[0] + ".mrc vesicles.mod")
        os.system("imodauto -f 3 -m 1 -h 0 " + os.path.splitext(os.path.split(self.last_mrc)[1])[0] + ".mrc vesicles.mod")
        os.system("imodjoin -c cell_outline.mod active_zone.mod merge1.mod")
        os.system("imodjoin -c merge1.mod vesicles.mod merge2.mod")
        handPick = False
        while yesOrNo:
            # these thing not general for all the files!!!!!
            os.system('imod -W '+self.file_name+' merge2.mod')
            handPick = mrc_cleaner.query_yes_no("Is the mod file ready?")
            if handPick:
                break
            else:
                yesOrNo = mrc_cleaner.query_yes_no("We can wait here for a while! do you want to pick by hand also?")

        # os.chdir('../..')
        print(os.getcwd())

        cytomod = 'cell_outline.mod'
        az = 'active_zone.mod'

        # sv = 'Results_Vesicles/20200615/bubu_elipsoid.mod'

        fullmod = 'merge.mod'

        tomogram = self.image_name
        nad = tomogram + '.nad'
        print(tomogram)

        cytomask = 'cytomask.mrc'
        az_mask = 'azmask.mrc'
        # sv_mask = './before_sphere.mrc'
        sv_mask = self.last_mrc

        # sv_mask_zoom_out = './Dummy_133_elipsoid_corrected_labels.mrc'
        # sv_mask_CV = 'mancorr_sv.mrc'

        labels_out = self.folder_to_save + 'labels.mrc'

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

        with mrcfile.new(labels_out, overwrite=True) as mrc:
            tomo = mrcfile.open(self.last_mrc)
            mrc.set_data(labels)
            mrc.voxel_size = tomo.voxel_size

        # double check
        labels = mrcfile.open(labels_out).data
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
        with mrcfile.new(labels_out, overwrite=True) as mrc:
            tomo = mrcfile.open(self.last_mrc)
            mrc.set_data(new_labels.astype(np.int16, copy=False))
            mrc.voxel_size = tomo.voxel_size
            mrc.header['origin'] = tomo.header['origin']



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.chdir('./data')
    dataset_dir = os.getcwd()
    datasetname_lst=os.listdir('.')
    dataset_lst=[dataset_dir+x for x in datasetname_lst]
    print("Which dataset are we working on?")
    [print(x) for x in datasetname_lst]
    print("?")
    choice = input()
    working_datset_index=datasetname_lst.index(choice)
    # print(working_datset_index)
    myPipeline=pipeline(datasetname_lst[working_datset_index])
    # for dataset in dataset_lst:
    #     pipeline(dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    myPipeline.run_deep()
    myPipeline.setup_prepyto_dir()
    myPipeline.zoom()
    myPipeline.outcell_remover()
    # myPipeline.thereshold_tunner()
    myPipeline.label_convexer()
    myPipeline.interactive_cleaner()
    myPipeline.sphere_vesicles()
    myPipeline.visualization_old_new()
    myPipeline.making_pyto_stuff()




















