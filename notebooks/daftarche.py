import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import prepyto

import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.registration
import numpy as np



dataset_directory = "/mnt/data/amin/Handpicked/115/"
dataset_directory = "/mnt/data/amin/ctrl/10"
# dataset_directory = "/mnt/data/amin/treatment/10"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
pl2.setup_prepyto_dir()
pl2.run_deep(force_run=True, rescale=1.0)
pl2.zoom()
# #
# # pl2.run_deep()
# # pl2.zoom()
# #
# ##### pl2.run_deep_at_multiple_rescale()
# #
ves_table= pl2.label_vesicles_simply(within_segmentation_region = False)
# # # print(ves_table)
# # #
# # #
#
# # delta_size =1
# # verysmall_vesicles = ves_table[(ves_table['area_zscore'] < -2)]
# # old_label= pl2.deep_labels.copy()
# # for i in verysmall_vesicles.iterrows():
# #     sub_old_label = old_label[verysmall_vesicles['bbox-0'][i[0]] - delta_size: verysmall_vesicles['bbox-3'][i[0]] + delta_size + 1,
# #                                        verysmall_vesicles['bbox-1'][i[0]] - delta_size: verysmall_vesicles['bbox-4'][i[0]] + delta_size + 1,
# #                                        verysmall_vesicles['bbox-2'][i[0]] - delta_size: verysmall_vesicles['bbox-5'][i[0]] + delta_size + 1]
# #     sub_old_label_mask = sub_old_label == verysmall_vesicles['label'][i[0]]
# #     pxj, pyj, pzj = np.where(sub_old_label_mask)
# #     sub_old_label[pxj,pyj,pzj] = 0
# # pl2.deep_labels = old_label.astype(np.uint16)
# # prepyto.save_label_to_mrc(pl2.deep_labels, pl2.deep_labels_path, template_path=pl2.image_path)
# #
# #
# #
# # collision_ves = ves_table[(ves_table['extent_zscore'] < -2) & (ves_table['area_zscore'] > .5)]
# # old_mask= pl2.deep_mask.copy()
# # old_label= pl2.deep_labels.copy()
# # for i in collision_ves.iterrows():
# #     print(i)
# #     sub_old_mask = old_mask[collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
# #                     collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
# #                     collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1]
# #
# #     sub_old_label = old_label[collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
# #                    collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
# #                    collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1]
# #
# #     sub_old_label_mask = sub_old_label != collision_ves['label'][i[0]]
# #     pxj, pyj, pzj = np.where(sub_old_label_mask)
# #     sub_old_mask[pxj,pyj,pzj] = 0
# #     thr, _ = prepyto.my_threshold(pl2.image, pl2.deep_mask)
# #     pre_labels, pre_nc = skimage.morphology.label(sub_old_mask > thr , return_num=True, connectivity=None)
# #     print(pre_nc)
# #     is_break=0
# #     for th in np.arange(thr, 1, 0.01):
# #         # temp=sub_old_label_mask>th
# #         temp = sub_old_mask > th
# #         labels, nc= skimage.morphology.label(temp, return_num = True , connectivity=None)
# #         if nc > pre_nc:
# #             is_break=1
# #             print(pre_nc,nc)
# #             px, py, pz = np.where(labels>0)
# #
# #             pxq, pyq, pzq = np.where(~sub_old_label_mask)
# #
# #             old_label[pxq + collision_ves['bbox-0'][i[0]] - delta_size,
# #                                 pyq + collision_ves['bbox-1'][i[0]] - delta_size,
# #                                 pzq + collision_ves['bbox-2'][i[0]] - delta_size] = 0
# #             # old_label[collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
# #             # collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
# #             # collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1] = labels
# #
# #             old_label[px + collision_ves['bbox-0'][i[0]] - delta_size,
# #                              py + collision_ves['bbox-1'][i[0]] - delta_size,
# #                              pz + collision_ves['bbox-2'][i[0]] - delta_size] = collision_ves['bbox-0'][i[0]] + 1000
# #             break
# #     if is_break==0:
# #         for th in np.arange(0.99, 1, 0.001):
# #             # temp=sub_old_label_mask>th
# #             temp = sub_old_mask > th
# #             labels, nc = skimage.morphology.label(temp, return_num=True, connectivity=None)
# #             if nc > pre_nc:
# #                 print(th)
# #                 is_break = 1
# #                 print(pre_nc, nc)
# #                 px, py, pz = np.where(labels > 0)
# #
# #                 pxq, pyq, pzq = np.where(~sub_old_label_mask)
# #
# #                 old_label[pxq + collision_ves['bbox-0'][i[0]] - delta_size,
# #                           pyq + collision_ves['bbox-1'][i[0]] - delta_size,
# #                           pzq + collision_ves['bbox-2'][i[0]] - delta_size] = 0
# #                 # old_label[collision_ves['bbox-0'][i[0]] - delta_size: collision_ves['bbox-3'][i[0]] + delta_size + 1,
# #                 # collision_ves['bbox-1'][i[0]] - delta_size: collision_ves['bbox-4'][i[0]] + delta_size + 1,
# #                 # collision_ves['bbox-2'][i[0]] - delta_size: collision_ves['bbox-5'][i[0]] + delta_size + 1] = labels
# #                 old_label[px + collision_ves['bbox-0'][i[0]] - delta_size,
# #                           py + collision_ves['bbox-1'][i[0]] - delta_size,
# #                           pz + collision_ves['bbox-2'][i[0]] - delta_size] = collision_ves['bbox-0'][i[0]] + 1000
# #                 break
# # #
# # #
# # # pl2.quick_setup(['deep_mask','deep_labels'])
# # old_label = skimage.morphology.label(old_label, connectivity=1)
# # old_label = old_label.astype(np.uint16)
# # pl2.deep_labels = old_label
# # prepyto.save_label_to_mrc(pl2.deep_labels, pl2.deep_labels_path, template_path=pl2.image_path)
# # # pl2.visualization_old_new('deep_labels', 'sphere_labels')
# #
# #
# # pl2.last_output_array_name= "deep_labels"
# #
# #
# # # # # pl2.threshold_tuner()
pl2.make_spheres()
# # # # #
# pl2.repair_spheres()


    # a= skimage.measure.regionprops_table(sub_old_label, properties=('centroid_local'))



# collision_ves = ves_table[(ves_table['extent_zscore'] < -2) & (ves_table['area_zscore'] > .5)]
# self.deep_labels = prepyto.remove_labels_under_points(self.deep_labels,collision_ves[['centroid-0', 'centroid-1', 'centroid-2']])
# pl2.automatic_evaluation()
# pl2.visualization_old_new('deep_labels', 'deep_mask')

# pl2.visualization_old_new('sphere_labels', 'deep_mask')
pl2.evaluation()
# tab= pl2.object_evaluation(reference_path='labels_out.mrc')