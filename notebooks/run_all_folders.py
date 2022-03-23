import os
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


dataset_directory = "/mnt/data/amin/Handpicked/"
# dataset_directory = "/mnt/data/amin/ctrl/"
# dataset_directory = "/mnt/data/amin/treatment/"
# dataset_directory = "/mnt/data/amin/bad/"
# dataset_directory = "
# /mnt/data/amin/bad/"


def my_function(directory):
    pl = prepyto.Pipeline(directory)
    pl.network_size = 64
    pl.setup_prepyto_dir()

    pl.run_deep(force_run=True, rescale=0.5)
    pl.zoom(force_run=True, )
    pl.label_vesicles(within_segmentation_region = True)
    pl.label_vesicles_simply(within_segmentation_region = True, input_array_name="deep_mask")
    pl.make_spheres()
    pl.repair_spheres()
    pl.clear_memory()
    res=pl.object_evaluation(reference_path='labels_out.mrc')
    # pl.make_full_modfile(input_array_name='convex_labels')
    # pl.make_full_label_file()
    return res


# directories = [os.path.abspath(x[0]) for x in os.walk(dataset_directory)]
folders= os.listdir(dataset_directory)
directories = [dataset_directory+x+'/' for x in folders]
# directories.remove(os.path.abspath(dataset_directory)) # don't want  main directory included

print("files")
print(directories)
all_res=[]
for i in directories:
  os.chdir(i)
  print(i)# Change working Directory
  res= my_function(i)

  all_res += [res]
  print(all_res)