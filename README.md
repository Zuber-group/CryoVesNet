# CryoVesNet
![Pipeline](images/github_figure.png)
## Installation

<pre> conda create -n cryoVesNet python=3.9</pre>
<pre>  conda activate cryoVesNet </pre>
<pre>  pip install -e . </pre>

## Requirements
<pre>
from setuptools import setup, find_packages

setup(
    name='prepyto',
    version='0.1.2',
    url='',
    packages=find_packages(),
    license='',
    author='Amin Khosrozadeh',
    author_email='',
    description='prepyto - branch pathlib',
    package_data={'': ['weights/weights.h5']},
    include_package_data=True,
    install_requires=['numpy<1.24', 'scikit-image<0.19', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari',
                      'keras','tensorflow<2.10'],

)
</pre>
## Using pre-trained model to segment cytoplasmic vesicles
To create the cytomask, you need to place cell_outline.mod file in the same directory as the tomogram.
You use the same script to build your pipeline, in case you are interested in all vesicles in tomogrmas you can set  in all vesicles within_segmentation_region = False.

<pre>
dataset_directory = "/mnt/data/amin/ctrl/"
pl = prepyto.Pipeline(directory)
pl.network_size = 64
pl.setup_prepyto_dir()
pl.run_deep(force_run=True, rescale=1.0)
pl.zoom(force_run=True, )
pl.label_vesicles(within_segmentation_region = True)
pl.label_vesicles_simply(within_segmentation_region = True, input_array_name="deep_mask")
pl.make_spheres()
pl.repair_spheres()
pl.make_full_modfile(input_array_name='convex_labels')
pl.make_full_label_file()
</pre>


## Folder Structure
<pre><font color="#268BD2"><b>.</b></font>
├── <font color="#859900"><b>az.mod</b></font>
├── <font color="#859900"><b>cell_outline.mod</b></font>
├── <font color="#268BD2"><b>deep</b></font>
│   ├── <font color="#D33682"><b>Dummy_133_trim.rec_processed.tiff</b></font>
│   ├── Dummy_133_trim.rec_segUnet.npy
│   ├── <font color="#D33682"><b>Dummy_133_trim.rec_segUnet.tiff</b></font>
│   └── <font color="#D33682"><b>Dummy_133_trim.rec_wreal_mask.tiff</b></font>
├── <font color="#859900"><b>Dummy_133_trim.rec</b></font>
├── <font color="#859900"><b>Dummy_133_trim.rec.nad *</b></font>
├── labels_out.mrc
├── <font color="#859900"><b>merge.mod</b></font>
├── new_labels_out.mrc
└── <font color="#268BD2"><b>prepyto</b></font>
    ├── Dummy_133_trim.rec_azmask.mrc
    ├── Dummy_133_trim.rec_clean_deep_labels.mrc
    ├── Dummy_133_trim.rec_convex_labels.mrc
    ├── Dummy_133_trim.rec_cytomask.mrc
    ├── Dummy_133_trim.rec_deep_labels.mrc
    ├── Dummy_133_trim.rec_final_vesicle_labels.mrc
    ├── Dummy_133_trim.rec_good.mrc
    ├── Dummy_133_trim.rec_sphere_dataframe.pkl
    ├── Dummy_133_trim.rec_sphere.mrc
    ├── Dummy_133_trim.rec_zoomed_mask.mrc
    ├── full_prepyto.mod
    ├── labels.mrc
    └── vesicles.mod

2 directories, 24 files</pre>
