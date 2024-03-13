import cryovesnet
import numpy as np
import napari
import tqdm
import skimage
import matplotlib.pyplot as plt


pl = cryovesnet.Pipeline("/Users/bzuber/Microscopic Anatomy Dropbox/Benoit Zuber/projects/deepvesicle/data/102_4e_trimmed")


pl.set_array('deep_labels')
pl.set_array('deep_mask')
pl.set_array('deep_winners_mask')


thresholded_maps = {}
for threshold in tqdm.tqdm(np.arange(0.8,1,0.04)):
    #thresholded_mask = np.where(pl.deep_mask>threshold, pl.deep_mask, 0)
    thresholded_winners = np.where(pl.deep_mask>threshold, pl.deep_winners_mask, 0) 
    thresholded_winners = np.round(thresholded_winners * 20).astype(np.int8)
    thresholded_maps[threshold] = thresholded_winners


with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(pl.image)
    #viewer.add_image(thresholded_mask)
    for threshold, thr_map in thresholded_maps.items():
        viewer.add_labels(thr_map, name=f"{threshold:.2f}")


diameter_dict = {}
for threshold in tqdm.tqdm(np.arange(0.8,1,0.01)):
    image_label = skimage.morphology.label(pl.deep_mask>threshold)
    area = skimage.measure.regionprops_table(image_label, properties=('bbox_area',))['bbox_area']
    diameter_dict[threshold] = area**(1/3)/pl.voxel_size


fig = plt.figure(figsize=(12,6))
fig.suptitle("diameter distribution")
_ = plt.hist(list(diameter_dict.values())[::5], label=list(diameter_dict.keys())[::5])
_ = plt.legend()


fig = plt.figure(figsize=(12,6))
fig.suptitle("diameter distribution")
_ = plt.hist(list(diameter_dict.values())[10::2], label=list(diameter_dict.keys())[10::2])
_ = plt.legend()


fig = plt.figure(figsize=(10,6))
fig.suptitle("diameter distribution")
_ = plt.hist(list(diameter_dict.values())[15:], label=list(diameter_dict.keys())[15:])
_ = plt.legend()


diameter_dict = {}
for threshold in tqdm.tqdm(np.arange(0.97,1,0.004)):
    image_label = skimage.morphology.label(pl.deep_mask>threshold)
    area = skimage.measure.regionprops_table(image_label, properties=('bbox_area',))['bbox_area']
    diameter_dict[threshold] = area**(1/3)/pl.voxel_size


fig = plt.figure(figsize=(10,6))
fig.suptitle("diameter distribution")
_ = plt.hist(list(diameter_dict.values())[::1], label=list(diameter_dict.keys())[::1])
_ = plt.legend()


fig = plt.figure(figsize=(10,6))
fig.suptitle("number of labels")
plt.xlabel("threshold")
plt.ylabel("number of labels")
plt.grid()
plt.scatter(x=list(diameter_dict.keys()), y=[len(v) for v in list(diameter_dict.values())])


pl.sphere_df.hist()


pl.identify_spheres_outliers(min_mahalanobis_distance=2)


pl.fix_spheres_interactively()


pl.visualization_old_new('sphere_labels','mancorr_labels')


pl.compute_sphere_dataframe('mancorr_labels')


pl.sphere_df.hist()


pl.identify_spheres_outliers()


pl.fix_spheres_interactively('mancorr_labels')


pl.compute_sphere_dataframe('mancorr_labels')


pl.identify_spheres_outliers()


pl.visualization_old_new('sphere_labels','mancorr_labels')


pl.make_full_modfile()
pl.make_full_label_file()
pl.initialize_pyto()



