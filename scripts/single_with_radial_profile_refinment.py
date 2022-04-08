import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import prepyto
from scipy.stats import chi2


dataset_directory = "/Users/bzuber/Dropbox/temp_only_copies/133/"
# dataset_directory = "/mnt/data/amin/ctrl/8"
#dataset_directory = "/mnt/data/amin/treatment/2"


pl2 = prepyto.Pipeline(dataset_directory)
pl2.network_size=64
#pl2.make_spheres(input_array_name='clean_deep_labels')
#we don't compute the spheres but use precomputed sphere_df (automatically loaded now)
mahalanobis_criteria = ['thickness','radius','membrane density']
print(f"CRITERIA = {mahalanobis_criteria}")
print('to fix:')
print(pl2.sphere_df[pl2.sphere_df.p < 0.3][['thickness','radius','membrane density','p']])
fixed, unfixed = pl2.refine_sphere_outliers('clean_deep_labels',mahalanobis_criteria=mahalanobis_criteria,
                                            p_threshold=0.3, drop_unfixed=True)
print('fixed:')
print(pl2.sphere_df.loc[fixed][['thickness','radius','membrane density','p']])
for i in range(10):
    print(f"CRITERIA = {mahalanobis_criteria}, i = {i}")
    print('to fix:')
    print(pl2.sphere_df[pl2.sphere_df.p < 0.3][['thickness', 'radius', 'membrane density', 'p']])
    fixed, unfixed = pl2.refine_sphere_outliers('clean_deep_labels',mahalanobis_criteria=mahalanobis_criteria,
                                                p_threshold=0.3, drop_unfixed=False)
    print('fixed:')
    print(pl2.sphere_df.loc[fixed][['thickness','radius','membrane density','p']])
    print('unfixed:')
    print(pl2.sphere_df.loc[unfixed][['thickness','radius','membrane density','p']])
    if fixed.empty: break

mahalanobis_criteria = ['thickness', 'membrane density','lumen/membrane density']
print(f"CRITERIA = {mahalanobis_criteria}")
pl2.sphere_df['mahalanobis'] = prepyto.mahalanobis_distances(pl2.sphere_df[mahalanobis_criteria])
pl2.sphere_df['p'] = 1 - chi2.cdf(pl2.sphere_df['mahalanobis'], len(mahalanobis_criteria))
print(pl2.sphere_df.loc[pl2.sphere_df.p < 0.3][['thickness','radius','membrane density','p']].sort_values('p'))

for i in range(10):
    print(f"CRITERIA = {mahalanobis_criteria}, i = {i}")
    print('to fix:')
    print(pl2.sphere_df[pl2.sphere_df.p < 0.3][['thickness', 'radius', 'membrane density', 'p']])
    fixed, unfixed = pl2.refine_sphere_outliers('clean_deep_labels',mahalanobis_criteria=mahalanobis_criteria,
                                                p_threshold=0.3, drop_unfixed=False)
    print('fixed:')
    print(pl2.sphere_df.loc[fixed][['thickness','radius','membrane density','p']])
    print('unfixed:')
    print(pl2.sphere_df.loc[unfixed][['thickness','radius','membrane density','p']])
    if fixed.empty: break

mahalanobis_criteria = ['thickness', 'membrane density','outer/membrane density']
for i in range(10):
    print(f"CRITERIA = {mahalanobis_criteria}, i = {i}")
    print('to fix:')
    print(pl2.sphere_df[pl2.sphere_df.p < 0.3][['thickness', 'radius', 'membrane density', 'p']])
    fixed, unfixed = pl2.refine_sphere_outliers('clean_deep_labels',mahalanobis_criteria=mahalanobis_criteria,
                                                p_threshold=0.3, drop_unfixed=False)
    print('fixed:')
    print(pl2.sphere_df.loc[fixed][['thickness','radius','membrane density','p']])
    print('unfixed:')
    print(pl2.sphere_df.loc[unfixed][['thickness','radius','membrane density','p']])
    if fixed.empty: break

print('the unfixed are mostly false positive, it is safe to use refine_sphere_outliers with drop_unfixed=True at this stage')