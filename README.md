# CryoVesNet

CryoVesNet is a deep learning-based method for automatic segmentation of synaptic vesicles in cryo-electron tomography (cryoET) data. It is based on a U-Net architecture trained on manually segmented tomograms and postprocessing steps. Notably, our method's ability to generalize across different datasets, namely from rat synaptosomes and to primary neuronal cultures, underscores its versatility and potential for widespread application. It is not restricted to synaptic vesicles but also can be applied to any spherical membrane-bound organelle. CryoVesNet is available as a Python package at this github repository.
You are also able to use this approch to segment any spherical vesicle or other organelles in any cell type.
Either, you can use pretrain network or train your own network based on interested dataset or use provided jupyter notebook to prepare your train dataset and train your network.
This package is developed and implemented @Zuber-group in Prof. Dr. Benoit Zuber's lab at the University of Bern, Switzerland.

## Installation
> **Attention:**
This program is presently in a developmental phase. Upcoming updates might introduce new functionalities, and significant modifications could occur across the existing and forthcoming versions, leading up to the stable version.
> 
You can install the package using conda and pip. After cloning the repository, you can install the package using the following commands:
1. Clone the repository
<pre> git clone https://github.com/Zuber-group/CryoVesNet/</pre>
2. Create a conda environment
<pre> conda create -n cryoVesNet python=3.9</pre>
3. Activate the conda environment
<pre>  conda activate cryoVesNet </pre>
4. Install the pre-requirmenets
<pre>  pip install -e . </pre>




## Requirements

[Click here for warning message](./warning-message.html)
> **Warning:**
We were using Linux build based ARM64 processors and to avoid using third party build and any confilct we used tensorflow<2.10 and numpy<1.24 to avoid any conflict. We are working on the new version of the project which will be compatible with all the latest version of the libraries.

<pre>
from setuptools import setup, find_packages

setup(
    name='prepyto',
    version='0.1.2',
    url='',
    packages=find_packages(),
    license='',
    author='Amin Khosrozadeh ',
    author_email='',
    description='prepyto - branch pathlib',
    package_data={'': ['weights/weights.h5']},
    include_package_data=True,
    install_requires=['numpy<1.24', 'scikit-image<0.19', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari',
                      'keras','tensorflow<2.10'],

)
</pre>
## Using the pre-trained model to segment cytoplasmic vesicles
To create the cytomask, you need to place cell_outline.mod file in the same directory as the tomogram.
You use the same script to build your pipeline, in case you are interested in all vesicles in tomogrmas you can set  in all vesicles within_segmentation_region = False.
We used object oreindted approach to build the pipeline. You can use the following [script](notebooks/single_dataset.py) to build your pipeline and run different steps of the pipeline.


1. Set the directory of the tomogram 
<pre>dataset_directory = "/mnt/data/tomogram_133/"</pre>
2. Creating the pipeline  
<pre> pl = prepyto.Pipeline(directory) </pre>
3. Set the network size (you can check other methods of the pipeline like check_files or prepare_deep) 
<pre> pl.network_size = 64 #the larger the better to avoid tiling effects </pre>
4. Setup the new directory  
<pre> pl.setup_prepyto_dir() </pre>
5. Run the deep learning network
<pre> pl.run_deep(force_run=True, rescale=1.0) </pre>
6. Zoom the mask to the original size of the tomogram
<pre> pl.zoom(force_run=True,) </pre>
7. Generate primary labels if you don not have cell_outline.mod file in the same directory as the tomogram you can set within_segmentation_region = False
<pre> pl.label_vesicles(within_segmentation_region = True) </pre>
8. Generate secondary labels
<pre> pl.label_vesicles_simply(within_segmentation_region = True, input_array_name="deep_mask") </pre>
9. Refinement using radial profile
<pre> pl.make_spheres() </pre>
10. Outlier detection and refinement
<pre> pl.repair_spheres() </pre>
11. This step ensures that the mod file is compatible with the pyto software
<pre> pl.make_full_modfile(input_array_name='convex_labels')
 pl.make_full_label_file() </pre>


## Folder Structure

The ** indicates the files that are needed for the pipeline to run.
The single * indicates the files that are necessary for removing the outer membrane vesicles and creating the mod files.
After running the pipeline, "deep" folder and "prepyto" folder will be created.
The "deep" folder contains the output of the deep learning network and the "prepyto" folder contains the output of the post-processing steps.


<pre><font color="#268BD2"><b>tomogram_133</b></font>
├── <font color="#859900"><b>az.mod</b></font>
├── <font color="#859900"><b>cell_outline.mod *</b></font>
├── <font color="#268BD2"><b>deep</b></font>
│   ├── <font color="#D33682"><b>Dummy_133_trim.rec_processed.tiff</b></font>
│   ├── Dummy_133_trim.rec_segUnet.npy
│   ├── <font color="#D33682"><b>Dummy_133_trim.rec_segUnet.tiff</b></font>
│   └── <font color="#D33682"><b>Dummy_133_trim.rec_wreal_mask.tiff</b></font>
├── <font color="#859900"><b>Dummy_133_trim.rec</b></font>
├── <font color="#859900"><b>Dummy_133_trim.rec.nad **</b></font>
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

## Interactive cleaning of the segmentation
You can use the interactive cleaning of the segmentation to remove false positive and adding false negative vesicles.
You can use the following command to run the interactive cleaning of the sepecific label file.

<pre>
pl.fix_spheres_interactively("final_vesicle_labels")
</pre>

## Train and create your own dataset
The notebooks allow to [generate the training data](notebooks/create_trainingset.ipynb) and [train the network](notebooks/training_vesicles.ipynb). Pre-trained weights for the network are provided in the weights folder.
## Schematic Workflow
![Pipeline](images/github_figure.png)
