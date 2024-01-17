import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import prepyto



dataset_directory = "/mnt/data/amin/Handpicked/133/"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()
pl2.run_deep(force_run=True, rescale=0.5)
pl2.zoom(force_run=True,)
pl2.label_vesicles(within_segmentation_region=True)
pl2.make_spheres('clean_deep_labels')
pl2.repair_spheres()
pl2.make_full_modfile(input_array_name='convex_labels')
pl2.make_full_label_file()
# res=pl2.object_evaluation(reference_path='labels_out.mrc')

# pl2.visualization_old_new("clean_deep_labels" , "sphere_labels")
# pl2.visualization_old_new("clean_deep_labels","convex_labels")