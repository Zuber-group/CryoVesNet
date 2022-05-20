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



dataset_directory = "/mnt/data/amin/Handpicked/84/"
# dataset_directory = "/mnt/data/amin/ctrl/8"
# dataset_directory = "/mnt/data/amin/treatment/5"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()

reference_path='labels_out.mrc'
reference_path = pl2.dir / reference_path
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
connected_vesicles = measures_pd[(measures_pd['extent'] < 0.5 )]
delta_size=1
# connected_vesicles = connected_vesicles.set_index('label')
# print( (connected_vesicles['label']))
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
    print(nc)





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
    print(np.unique(labels))

reff= skimage.measure.label(reff, connectivity=1)
visualization.viz_labels(pl2.image, [reff, orig_reff], ['repaired', 'manual'])



