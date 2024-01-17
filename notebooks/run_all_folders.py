# This script runs the pipeline on all folders in a directory

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import prepyto

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


def single_dataset_handler(directory):
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
    # res=pl.object_evaluation(reference_path='labels_out.mrc')
    pl.make_full_modfile(input_array_name='convex_labels')
    pl.make_full_label_file()
    # return res


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
  # res= my_function(i)
  res = single_dataset_handler(i)
  # all_res += [res]
  # print(all_res)