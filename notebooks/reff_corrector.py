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
import pandas as pd
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max



dataset_directory = "/mnt/data/amin/Handpicked/"
# dataset_directory = "/mnt/data/amin/ctrl/"
# dataset_directory = "/mnt/data/amin/treatment/"

folders= os.listdir(dataset_directory)
directories = [dataset_directory+x+'/' for x in folders]
# directories.remove(os.path.abspath(dataset_directory)) # don't want  main directory included

print("files")
print(directories)
directories=["/mnt/data/amin/cleaned/1/"]
t=0
for j in directories:
    os.chdir(j)
    print(j)# Change working Directory
    pl2 = prepyto.Pipeline(j)
    pl2.network_size=64
    pl2.setup_prepyto_dir()

    reference_path='31_rotx-bin-trim.nad.rec_final_vesicle_labels.mrc'
    result_path='31_rotx-bin-trim.nad.rec_final_vesicle_labels.mrc'
    reference_path = pl2.save_dir / reference_path
    reference = mrcfile.open(reference_path)
    reff = reference.data

    pl2.set_array('image')
    pl2.set_array('cytomask')


    reff = reff * pl2.cytomask
    reff = reff.astype(np.uint16)
    temp= np.where(reff<10)
    reff[temp]=0
    orig_reff= reff.copy()
    measures = skimage.measure.regionprops_table(reff, properties=('label','extent','bbox','area'))
    measures_pd = pd.DataFrame(measures)
    connected_vesicles = measures_pd[(measures_pd['extent'] < 0.44 )]
    t=t+ int(connected_vesicles.shape[0])
    delta_size=1

    # connected_vesicles = connected_vesicles.set_index('label')
    print( (connected_vesicles['label']))
    for i, row in connected_vesicles.iterrows():
        sub_old_label = reff[
                        connected_vesicles['bbox-0'][i] - delta_size: connected_vesicles['bbox-3'][i] + delta_size + 1,
                        connected_vesicles['bbox-1'][i] - delta_size: connected_vesicles['bbox-4'][i] + delta_size + 1,
                        connected_vesicles['bbox-2'][i] - delta_size: connected_vesicles['bbox-5'][i] + delta_size + 1]
        # sub_new_label = prediction[
        #                 measures['bbox-0'][i] - delta_size: measures['bbox-3'][i] + delta_size + 1,
        #                 measures['bbox-1'][i] - delta_size: measures['bbox-4'][i] + delta_size + 1,
        #                 measures['bbox-2'][i] - delta_size: measures['bbox-5'][i] + delta_size + 1]


        # reff[pxj, pyj, pzj] = 0
        pxj, pyj, pzj = np.where(sub_old_label != connected_vesicles['label'][i])
        temp = sub_old_label[pxj, pyj, pzj]
        sub_old_label_mask = sub_old_label == connected_vesicles['label'][i]

        measure_label,nc = skimage.measure.label(sub_old_label_mask,return_num=True)
        # , properties = ('label', 'extent', 'bbox', 'centroid')
        # print(nc)





        distance = ndi.distance_transform_edt(sub_old_label_mask)
        coords = peak_local_max(distance, footprint=np.ones((7, 7,7 )), labels=sub_old_label_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=sub_old_label_mask)

        # measure_label, nc = skimage.measure.label(sub_old_label_mask, return_num=True)
        # , properties = ('label', 'extent', 'bbox', 'centroid')

        labels[pxj, pyj, pzj] = temp
        reff[
        connected_vesicles['bbox-0'][i] - delta_size: connected_vesicles['bbox-3'][i] + delta_size + 1,
        connected_vesicles['bbox-1'][i] - delta_size: connected_vesicles['bbox-4'][i] + delta_size + 1,
        connected_vesicles['bbox-2'][i] - delta_size: connected_vesicles['bbox-5'][i] + delta_size + 1] = labels
        # print(np.unique(labels))

    reff= skimage.measure.label(reff, connectivity=3)
    reff = reff.astype(np.uint16)

    measures_x = skimage.measure.regionprops_table(reff, properties=('label','extent','bbox','area'))
    measures_pd_x = pd.DataFrame(measures_x)
    connected_vesicles_x = measures_pd_x[(measures_pd_x['extent'] < 0.44 )]
    print((connected_vesicles_x['label']))
    if((connected_vesicles_x.shape[0])==0):
        print("Solved!")
        # visualization.viz_labels(pl2.image, [reff, orig_reff], ['repaired', 'manual'])
        prepyto.save_label_to_mrc(reff, j+result_path,template_path=pl2.image_path)
    # prepyto.save_label_to_mrc(reff, j + result_path, template_path=pl2.image_path)
