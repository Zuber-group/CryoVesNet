# This script runs the pipeline on single tomograms

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cryovesnet

dataset_directory = "/media/amin/mtwo/Handpicked/116"


pl2 = cryovesnet.Pipeline(dataset_directory,pattern='*.rec.nad')
pl2.setup_cryovesnet_dir(initialize=False, make_masks=False)
#
#

# pl2.run_deep(force_run=True, gauss=True, rescale= None,  weight_path=None)
pl2.run_deep(force_run=False, gauss=False, rescale= None, augmentation_level=4,  weight_path=None)
# pl2.rescale(force_run=True,slice_range=[25,95])
pl2.rescale(force_run=False, slice_range=None)
pl2.label_vesicles(input_array_name="deep_mask", within_segmentation_region=False,threshold_coef=None)
pl2.label_vesicles_adaptive(expanding = False, convex=True, separating=True)
df = pl2.make_spheres(tight=True, keep_ellipsoid=True)
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

