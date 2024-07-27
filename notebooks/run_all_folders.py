# This script runs the pipeline on all folders in a directory

import warnings
import os
import glob
import shutil

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import cryovesnet


# Set up the directory where the folders are like this:
# Handpicked
# ├── 102
# ├── 114
# ├── 115
# ├── 116
# ├── 123
# ├── 128
# ├── 132
# ├── 133
# ├── 134
# ├── 73
# ├── 80
# └── 84
dataset_directory = "/mnt/data/amin/Handpicked/"
# dataset_directory = "/mnt/data/amin/ctrl/"
# dataset_directory = "/mnt/data/amin/treatment/"
# dataset_directory = "/media/amin/mtwo/ctrl/"
# dataset_directory = "/media/amin/mtwo/treatment/"

def single_dataset_handler(directory,path_to_model='../weights/weights.h5',gauss=True):
    pl = cryovesnet.Pipeline(directory)
    pl.setup_cryovesnet_dir()
    pl.run_deep(force_run=True, rescale=None, gauss=gauss,augmentation_level=1, weight_path=path_to_model)
    pl.zoom(force_run=True, )
    pl.label_vesicles(within_segmentation_region = True,threshold_coef=None)
    pl.label_vesicles_simply( expanding = True,convex=False,separating=True)
    pl.make_spheres()
    pl.repair_spheres()
    pl.make_full_modfile(input_array_name='convex_labels')
    pl.make_full_label_file()
    return



# directories = [os.path.abspath(x[0]) for x in os.walk(dataset_directory)]
folders= os.listdir(dataset_directory)
directories = [dataset_directory+x+'/' for x in folders]
# directories.remove(os.path.abspath(dataset_directory)) # don't want  main directory included

print("Files:")
print(directories)
all_res=[]
for i in directories:
  os.chdir(i)
  print(i)# Change working Directory
  res = single_dataset_handler(i)
