import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import prepyto

import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.registration
import numpy as np



dataset_directory = "/mnt/data/amin/Handpicked/102/"
# dataset_directory = "/mnt/data/amin/ctrl/10"
# dataset_directory = "/mnt/data/amin/treatment/10"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()
pl2.run_deep(force_run=True, rescale=0.5)
pl2.zoom(force_run=True,)
# pl2.label_vesicles(within_segmentation_region=True)
pl2.label_vesicles_simply(within_segmentation_region = True)
pl2.make_spheres()
pl2.repair_spheres()
pl2.object_evaluation()