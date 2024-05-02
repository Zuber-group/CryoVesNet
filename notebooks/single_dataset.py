# This script runs the pipeline on single tomograms

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cryovesnet

dataset_directory = "/mnt/data/amin/cleaned/2new/"
# dataset_directory = "/home/amin/Pictures/ForAmin/wild_overlap/"
dataset_directory = "/media/amin/mtwo/133/"

pl2 = cryovesnet.Pipeline(dataset_directory)
pl2.setup_cryovesnet_dir()


pl2.run_deep(force_run=True, rescale=None,weight_path='/mnt/data/Amin/Data/training_logs/20240429-192102_train_dataset_32_synaptasome_32/weights_best_loss.h5')
pl2.zoom(force_run=True)
pl2.label_vesicles(within_segmentation_region=True)
pl2.label_vesicles_simply(expanding = False, input_array_name="deep_mask")
pl2.make_spheres('deep_labels')
pl2.repair_spheres()
# pl2.make_full_modfile(input_array_name='convex_labels')
# pl2.make_full_label_file()

res=pl2.object_evaluation(reference_path='labels_out.mrc')

# pl2.visualization_old_new("clean_deep_labels" , "sphere_labels")
# pl2.visualization_old_new("clean_deep_labels","convex_labels")



# pl2.set_array('image')
# pl2.set_array("cytomask")
# pl2.set_array("deep_mask")
# pl2.set_array("deep_labels")
# pl2.set_array("convex_labels")
# pl2.last_output_array_name="convex_labels"
# pl2.fix_spheres_interactively()
#
# pl2.fix_spheres_interactively("mancorr_labels")

