import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import prepyto
from prepyto import visualization
import mrcfile
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.registration
import numpy as np



dataset_directory = "/mnt/data/amin/Handpicked/123/"
dataset_directory = "/mnt/data/amin/ctrl/1"
# dataset_directory = "/mnt/data/amin/treatment/6"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()

# pl2.set_array('image')
# pl2.set_array('deep_mask')
# pl2.set_array('deep_labels')
# pl2.set_array('clean_deep_labels')
# pl2.set_array('sphere_labels')
# pl2.set_array('convex_labels')

# pl2.run_deep(force_run=True, rescale=1.0)
# pl2.zoom(force_run=True,)
# ## pl2.label_vesicles(within_segmentation_region=True)
ves=pl2.label_vesicles_simply(within_segmentation_region = True)
pl2.make_spheres()
pl2.repair_spheres(m=10)
pl2.object_evaluation(reference_path='labels_out.mrc')
# pl2.evaluation()
# pl2.visualization_old_new("deep_labels","convex_labels")
# t=pl2.object_evaluation()
# # #
# pl2.set_array('image')
# pl2.set_array('deep_mask')
# pl2.set_array('deep_labels')
# pl2.set_array('clean_deep_labels')
# pl2.set_array('sphere_labels')
# pl2.set_array('convex_labels')
#
#
# reference_path='labels_out.mrc'
# reference_path = pl2.dir / reference_path
# reference = mrcfile.open(reference_path)
# mask = reference.data
# mask = mask >= 10
# visualization.viz_labels(pl2.image, [pl2.deep_labels,pl2.clean_deep_labels,pl2.sphere_labels,pl2.convex_labels,mask],['deep','clean deep','sphere','repaired','manual'])
# # #


# p=-1
# pl2.make_full_modfile(input_array_name='convex_labels',q=p)
# pl2.make_full_label_file(q=p)
# res=pl2.object_evaluation(reference_path='labels_out.mrc')

# pl2.visualization_old_new("clean_deep_labels" , "sphere_labels")
# pl2.visualization_old_new("clean_deep_labels","convex_labels")