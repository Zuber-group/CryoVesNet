# This script runs the pipeline on single tomograms

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cryovesnet

dataset_directory = "/mnt/data/amin/cleaned/2new/"
# dataset_directory = "/home/amin/Pictures/ForAmin/wild_overlap/"

pl2 = cryovesnet.Pipeline(dataset_directory)
pl2.setup_cryovesnet_dir()
pl2.run_deep(force_run=True, rescale=None,weight_path='/media/amin/mtwo/train2d/20240501-071015_train_dataset_2d_128_synaptasome_1000_unet/weights_best_loss.h5')
pl2.zoom(force_run=True)
pl2.label_vesicles(within_segmentation_region=True)
pl2.label_vesicles_simply(expanding = False, input_array_name="deep_mask")
pl2.make_spheres('clean_deep_labels')
pl2.repair_spheres()
# pl2.make_full_modfile(input_array_name='convex_labels')
# pl2.make_full_label_file()

res=pl2.object_evaluation(reference_path='labels_out.mrc')

# pl2.visualization_old_new("clean_deep_labels" , "sphere_labels")
# pl2.visualization_old_new("clean_deep_labels","convex_labels")
