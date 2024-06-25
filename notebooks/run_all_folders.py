# This script runs the pipeline on all folders in a directory

import warnings
import os
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import shutil

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Here you can choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import cryovesnet

# dataset_directory = "/media/amin/mtwo/Handpicked/"
# dataset_directory = "/media/amin/mtwo/ctrl/"
# dataset_directory = "/media/amin/mtwo/treatment/"
# dataset_directory = ["/media/amin/mtwo/Handpicked/" , "/media/amin/mtwo/ctrl/" , "/media/amin/mtwo/treatment/"]
dataset_directory = ["/media/amin/mtwo/Handpicked/" ]
# dataset_directory = ["/media/amin/mtwo/ctrl/" ]
# dataset_directory = ["/media/amin/mtwo/treatment/"]

def single_dataset_handler(directory,path_to_model='../weights/weights_v1.h5',gauss=True):
    pl = cryovesnet.Pipeline(directory)
    pl.setup_cryovesnet_dir()
    # weight_path = '/mnt/data/Amin/Data/training_logs/20240507-164351_train_dataset_32_synaptasome_1024_0.0_32/weights_best_loss.h5'
    pl.run_deep(force_run=True, rescale=None, gauss=gauss,augmentation_level=1, weight_path=path_to_model)
    # pl.run_deep(force_run=True, rescale=None,
    #             path_to_model='/home/amin/PycharmProjects/CryoVesNetNEW/cryovesnet/weights/weights_old.h5')
    #pl.run_membrain(path_to_model=path_to_model,force_run=True)
    pl.zoom(force_run=True, )
    pl.label_vesicles(input_array_name='deep_mask',within_segmentation_region = True,threshold_coef=None)
    pl.label_vesicles_simply( expanding = True,convex=False,sperating=True)
    pl.make_spheres()
    pl.repair_spheres()
    pl.clear_memory()
    res=pl.object_evaluation(reference_path='labels_out.mrc')
    with open('results_final.txt', 'a') as file:
        file.write('/'.join(path_to_model.split('/')[-2:])+str(gauss)+"_added4,")
        file.write(",".join(map(str, res[:5])) + "\n")
    old_folder_name = 'cryovesnet'
    new_folder_name = 'cryovesnet_' + path_to_model.split('/')[-2]+'_added4_'+str(gauss)
    # new_folder_name = 'cryovesnet_membrain'
    if os.path.exists(new_folder_name):
        print(f"The directory {new_folder_name} already exists. Removing it.")
        shutil.rmtree(new_folder_name)

    os.rename(old_folder_name, new_folder_name)

    # pl.make_full_modfile(input_array_name='convex_labels')
    # pl.make_full_label_file()
    return res


# directories = [os.path.abspath(x[0]) for x in os.walk(dataset_directory)]
# folders= os.listdir(dataset_directory)
# directories = [dataset_directory+x+'/' for x in folders if not '.txt' in x]
# directories.remove(os.path.abspath(dataset_directory)) # don't want  main directory included


training_dir = "/mnt/data/Amin/Data/training_logs"
training_dir = '/mnt/data/Amin/Data_latest/training_logs/'

training_dir = '/mnt/data/Amin/Data_latest/training_logs/20240505-224125_train_dataset_32_synaptasome_32_0.2/'
# training_dir = '/mnt/data/Amin/Data_latest/2d_training_logs'

training_dir = '/mnt/data/Amin/Data/training_logs/20240507-164626_train_dataset_32_synaptasome_1024_0.2_32'

training_dir = '/mnt/data/Amin/Data_latest/training_logs/20240429-224708_train_dataset_64_synaptasome_8000_64'
# Use glob to find all .h5 files recursively within all subdirectories
file_paths = glob.glob(os.path.join(training_dir, '**', '*.h5'), recursive=True)
# file_paths = [f for f in file_paths if '1axes_2d_64' in f]
# file_paths = [f for f in file_paths if '_best_loss' in f]
# file_paths =[f for f in file_paths if 'resnet' not in f]
# file_paths =[f for f in file_paths if 'inception' not in f]
# file_paths = ['~/membrain-seg/newsetup/checkpoints/membrain-seg_v0_1-epoch=999-val_loss=0.60.ckpt']
# file_paths = ['/home/amin/membrain-seg/checkpoints/checkpoints/membrain-seg_v0_1-epoch=999-val_loss=0.52.ckpt']
# file_paths = ['/mnt/data/Amin/Data_latest/training_logs/20240502-113341_train_dataset_64_synaptasome_8000_64/weights_best_loss.h5']
# file_paths = ['/mnt/data/Amin/Data_latest/training_logs/20240505-224125_train_dataset_32_synaptasome_32/weights_best_loss.h5']

# file_paths = ['/home/amin/PycharmProjects/CryoVesNetNEW/cryovesnet/weights/weights_old.h5']
# file_paths = ['/home/amin/membrain-seg/checkpoints/checkpoints/membrain-seg_v0_1-epoch=999-val_loss=0.52.ckpt']
# training_dir = '/mnt/data/Amin/Data_latest/training_logs/20240429-224708_train_dataset_64_synaptasome_8000_64'
# file_paths += glob.glob(os.path.join(training_dir, '**', '*.h5'), recursive=True)

# file_paths = ['/home/amin/PycharmProjects/CryoVesNetNEW/cryovesnet/weights/weights.h5']

# file_paths = ['/mnt/data/Amin/Data_latest/training_logs/20240429-224708_train_dataset_64_synaptasome_8000_64']

print(file_paths)
for d in dataset_directory:
    folders = os.listdir(d)
    directories = [d + x + '/' for x in folders if not '.txt' in x]
    for gauss in [False,True]:
        for model_path in file_paths:
            print("Files:")
            print(directories)
            all_res=[]
            errors = []
            for i in directories:
                os.chdir(i)
                print(i)# Change working Directory
                # res= my_function(i)
                try:
                    res = single_dataset_handler(i,path_to_model=model_path,gauss=gauss)
                    # true_labels_list.append(true)
                    # predicted_probs_list.append(predict)
                except Exception as e:
                    error_message = f"Error in {i}: {e}"
                    print(error_message)
                    errors.append(i)
                    res=[0,0,0,0,0]

                all_res += [res]
                # print(f"Error in {i}")
                # errors.append(i)
            os.chdir(d)
            with open('_'.join(model_path.split('/')[-2:])+'_final_added4_'+str(gauss)+'.txt', 'w') as file:
                for directory, sublist in zip(directories, all_res):
                    file.write(directory + "," + ",".join(map(str, sublist)) + "\n")
          # res = single_dataset_handler(i)
          # all_res += [res]
          # print(all_res)



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
