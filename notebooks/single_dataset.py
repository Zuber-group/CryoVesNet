import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import prepyto

import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.registration
import numpy as np



dataset_directory = "/mnt/data/amin/Handpicked/84/"
# dataset_directory = "/mnt/data/amin/ctrl/8"
dataset_directory = "/mnt/data/amin/treatment/3"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()
pl2.set_array('deep_labels')
pl2.set_array('image')
pl2.set_array('convex_labels')
# pl2.run_deep(force_run=True, rescale=1.0)
# pl2.zoom(force_run=True,)
# ## pl2.label_vesicles(within_segmentation_region=True)
pl2.label_vesicles_simply(within_segmentation_region = True)
pl2.make_spheres('deep_labels')
pl2.repair_spheres(m=4)
# pl2.evaluation()
# pl2.visualization_old_new("sphere_labels","convex_labels")

# p=-1
# pl2.make_full_modfile(input_array_name='convex_labels',q=p)
# pl2.make_full_label_file(q=p)
res=pl2.object_evaluation(reference_path='labels_out.mrc')