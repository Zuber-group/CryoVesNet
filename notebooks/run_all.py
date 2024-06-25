# This script runs the pipeline on all folders in a directory

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import cryovesnet

dataset_directory = "/mnt/data/amin/Handpicked/"
dataset_directory = "/media/amin/mtwo/Handpicked/"
# dataset_directory = "/mnt/data/amin/ctrl/"
# dataset_directory = "/mnt/data/amin/treatment/"
# dataset_directory = "/media/amin/mtwo/treatment/"


def single_dataset_handler(directory):
    pl = cryovesnet.Pipeline(directory)
    pl.setup_cryovesnet_dir()
    pl.run_deep(force_run=True, rescale=None, weight_path='/mnt/data/Amin/Data/training_logs/20240502-113341_train_dataset_64_synaptasome_8000_64/weights_best_dice.h5')
    pl.zoom(force_run=True, )
    pl.label_vesicles(within_segmentation_region = True)
    pl.label_vesicles_simply(expanding = False,input_array_name="deep_mask")
    pl.make_spheres('clean_deep_labels')
    pl.repair_spheres()
    pl.clear_memory()
    res=pl.object_evaluation(reference_path='labels_out.mrc')
    with open('results.txt', 'a') as file:
        file.write(" ".join(map(str, res)) + "\n")
    pl.make_full_modfile(input_array_name='convex_labels')
    # pl.make_full_label_file()
    return res


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
  all_res += [res]
  # print(all_res)