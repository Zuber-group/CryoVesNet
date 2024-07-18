# This script runs the pipeline on single tomograms
from pathlib import Path
import os

# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cryovesnet




dataset_directory = Path("E:/Amin_Khosrozadeh/emd_30365")



pl2 = cryovesnet.Pipeline(dataset_directory,pattern='*.bin')
pl2.setup_cryovesnet_dir(initialize=False, make_masks=False)
#
#

# pl2.run_deep(force_run=True, gauss=True, rescale= None,  weight_path=None)
pl2.run_deep(force_run=False, gauss=False, rescale= None, augmentation_level=4,  weight_path=None)
pl2.zoom(force_run=True,slice_range=[1,130])
# pl2.zoom(force_run=True,slice_range=None)
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
# pl2.last_output_array_name= "clean_deep_labels"
# pl2.fix_spheres_interactively()
# # pl2.repair_spheres()
pl2.fix_spheres_interactively(max_expected_diameter=45)
#
# pl2.fix_spheres_interactively("mancorr_labels")

