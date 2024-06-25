# This script runs the pipeline on single tomograms

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import cryovesnet

dataset_directory = "/mnt/data/amin/cleaned/2new/"
dataset_directory = "/mnt/data/amin/cleaned/133/"
dataset_directory = "/media/amin/mtwo/Handpicked/116"
# dataset_directory = "/home/amin/Pictures/emd_12727/"
# # dataset_directory = "/home/amin/Pictures/ForAmin/wild_overlap/"
# dataset_directory = "/media/amin/mtwo/treatment/10/"
# # dataset_directory = "/home/amin/Pictures/ForAmin/wild/"
# # dataset_directory = "/home/amin/Pictures/ForAmin/SynapsePos70_NAD_10_16"
# dataset_directory = "/media/amin/mtwo/Handpicked/133"
# dataset_directory = "/home/amin/Pictures/manuel/droplet/"
dataset_directory = "/home/amin/Pictures/g4_MMM1_TS5_dose-filt/"
# dataset_directory = "/home/amin/Pictures/g5_MMM1_TS2_dose-filt"
# dataset_directory = "/home/amin/Pictures/g4_MMM1_TS3_dose-filt"
# dataset_directory = "/home/amin/Pictures/emd_13761"
# dataset_directory = "/home/amin/Pictures/emd_10409"
# # dataset_directory = "/home/amin/Pictures/emd_3977"
dataset_directory = "/home/amin/Pictures/emd_30364"
# dataset_directory = "/home/amin/Pictures/isonet"
# dataset_directory = "/home/amin/Pictures/WT_01"
# dataset_directory = "/home/amin/Pictures/emd_15474"
# dataset_directory = "/home/amin/Pictures/emd_4869"
# dataset_directory = "/home/amin/Pictures/emd_12727"


pl2 = cryovesnet.Pipeline(dataset_directory,pattern='*.rec.nad')
pl2.setup_cryovesnet_dir(initialize=False, make_masks=False)
#
#

# pl2.run_deep(force_run=True, gauss=True, rescale= None,  weight_path=None)
pl2.run_deep(force_run=False, gauss=False, rescale= 1.2, augmentation_level=4,  weight_path='/mnt/data/Amin/Data_latest/training_logs/20240505-224125_train_dataset_32_synaptasome_32_0.2/weights_best_loss.h5')
# pl2.run_deep(force_run=True, gauss=True, rescale= None, augmentation_level=1,  weight_path='/mnt/data/Amin/Data_latest/2d_training_logs/20240524-135023_train_dataset_1axes_2d_064_synaptasome_128_eman2/weights_best_dice.h5')
# pl2.run_deep(force_run=True, gauss=False, rescale= None,  weight_path='/mnt/data/Amin/Data_latest/training_logs/20240527-154714_train_dataset_32_synaptasome_0.0_32/weights_best_dice.h5')
# pl2.run_deep(force_run=True, rescale=None,weight_path='/mnt/data/Amin/Data/training_logs/20240429-192507_train_dataset_64_nad_neuron_64/weights_best_dice.h5')
# pl2.run_membrain(path_to_model='/home/amin/membrain-seg/checkpoints/checkpoints/membrain-seg_v0_1-epoch=999-val_loss=0.52.ckpt' ,force_run=True)
# pl2.zoom(force_run=True,slice_range=(30,200))
# pl2.zoom(force_run=True,slice_range=[25,95])
pl2.zoom(force_run=False,slice_range=None)
pl2.label_vesicles(input_array_name="deep_mask", within_segmentation_region=False,threshold_coef=0.7)
pl2.label_vesicles_simply( expanding = False,convex=True,sperating=True)
df = pl2.make_spheres(tight=True, keep_elipsoid=True)
pl2.repair_spheres(m=4)
# # # pl2.make_full_modfile(input_array_name='convex_labels')
# # # pl2.make_full_label_file()
# #
# res=pl2.object_evaluation(reference_path='labels_out.mrc')
#
# pl2.visualization_old_new("clean_deep_labels" , "sphere_labels")
# pl2.visualization_old_new("clean_deep_labels","convex_labels")
#

#
# pl2.set_array('image')
# # pl2.set_array("cytomask")
# # pl2.set_array("deep_mask")
# # pl2.set_array("deep_labels")
# pl2.set_array("convex_labels")
pl2.last_output_array_name="convex_labels"
# pl2.last_output_array_name="sphere_labels"
# pl2.last_output_array_name= "mancorr_labels"
# pl2.last_output_array_name="deep_labels"
# pl2.fix_spheres_interactively()
# # pl2.repair_spheres()
pl2.fix_spheres_interactively(max_expected_diameter=45)
#
# pl2.fix_spheres_interactively("mancorr_labels")

