# This script runs the pipeline on single tomograms
from pathlib import Path
import os
import cProfile
import pstats



# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cryovesnet



dataset_directory = Path("/media/amin/mtwo/84")
# dataset_directory = Path("E:/Amin_Khosrozadeh/134")




pl2 = cryovesnet.Pipeline(dataset_directory,pattern='*.rec.nad')
pl2.setup_cryovesnet_dir(initialize=False, make_masks=True)
#
#

# pl2.run_deep(force_run=True, gauss=True, rescale= None,  weight_path=None)
# pl2.run_deep(force_run=True,augmentation_level=4)
pl2.run_deep(force_run=True,augmentation_level=1)
# pl2.run_deep(force_run=True, gauss=False, rescale= None, augmentation_level=4,  weight_path=None)
pl2.rescale(force_run=True, slice_range=None)
#pl2.rescale(force_run=True,slice_range=[1,130])

pl2.label_vesicles(input_array_name="deep_mask", within_segmentation_region=True,threshold_coef=None)
# pl2.label_vesicles(input_array_name="deep_mask", within_segmentation_region=False,threshold_coef=0.9)

pl2.label_vesicles_adaptive(expanding = False, convex=False, separating=True)
df = pl2.make_spheres(tight=False, keep_ellipsoid=False)
pl2.repair_spheres()


#pl2.make_full_modfile(input_array_name='convex_labels')

### For manual correction ###
pl2.last_output_array_name="convex_labels"
pl2.fix_spheres_interactively(max_expected_diameter=45)


### In case you want to continue the manual correction ###
# pl2.last_output_array_name= "mancorr_labels"
# pl2.fix_spheres_interactively()
### or:
# pl2.fix_spheres_interactively("mancorr_labels")


### To clean the old files ###
#os.system('rm -rf /media/amin/mtwo/84/pyto')

## In case u jsut want the automatic segmentation without manual correction
## pl2.last_output_array_name="convex_labels"

### For connector segmentation ###
# pl2.make_full_modfile()
# pl2.make_full_label_file()
# pl2.pyto_wrapper()
