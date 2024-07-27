from pathlib import Path
import os
import cProfile
import pstats


# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cryovesnet

dataset_directory = Path("E:/Amin_Khosrozadeh/emd_30365")

def run_pipeline():
    dataset_directory = Path("E:/Amin_Khosrozadeh/emd_30365")

    pl2 = cryovesnet.Pipeline(dataset_directory, pattern='*.bin')
    pl2.setup_cryovesnet_dir(initialize=False, make_masks=False)

    pl2.run_deep(force_run=True, gauss=True, rescale=None, augmentation_level=1, weight_path=None)
    pl2.rescale(force_run=True, slice_range=[1, 130])
    pl2.label_vesicles(input_array_name="deep_mask", within_segmentation_region=False, threshold_coef=0.7)
    pl2.label_vesicles_adaptive(expanding=True, convex=True, sperating=True)
    df = pl2.make_spheres(tight=True, keep_ellipsoid=True)
    pl2.repair_spheres(m=4)


if __name__ == "__main__":
    # Run the profiler
    cProfile.run('run_pipeline()', 'pipeline_stats')

    # Print the stats
    p = pstats.Stats('pipeline_stats')
    print("\nTop 30 functions by cumulative time:")
    p.sort_stats('cumulative').print_stats(30)

    print("\nTop 30 functions by internal time:")
    p.sort_stats('time').print_stats(30)

    print("\nCallers for the top 30 functions:")
    p.print_callers(30)