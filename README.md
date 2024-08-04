# CryoVesNet

CryoVesNet is a deep learning-based method for automatic segmentation of synaptic vesicles in cryo-electron tomography (cryoET) data. It is based on a U-Net architecture trained on manually segmented tomograms and postprocessing steps. Notably, our method's ability to generalize across different datasets, namely from rat synaptosomes and to primary neuronal cultures, underscores its versatility and potential for widespread application. It is not restricted to synaptic vesicles but can also be applied to any spherical membrane-bound organelle.
You can either use a pre-train network or use the provided jupyter notebook to prepare your train dataset and train your network.
This package is developed and implemented [@Zuber-group](https://github.com/Zuber-group) in Benoit Zuber's lab at the University of Bern, Switzerland.

## Installation
> **Attention:**
This program is presently in a developmental phase. Upcoming updates might introduce new functionalities, and significant modifications could occur across the existing and forthcoming versions, leading up to the stable version.
> 
You can install the package using conda and pip. After cloning the repository, you can install the package using the following commands:
1. Clone the repository
<pre> git clone https://github.com/Zuber-group/CryoVesNet/</pre>
2. Create a conda environment
<pre> conda create -n CryoVesNet python=3.9</pre>
3. Activate the conda environment
<pre>  conda activate CryoVesNet </pre>
4. Install the pre-requirements
<pre>  pip install -e . </pre>


> **Warning:**
We were using Linux build-based ARM64 processors and to avoid using third-party build and any conflict we used tensorflow<2.10 to avoid any conflict. We are working on the new version of the project which will be compatible with all the latest versions of the libraries.

> **OS Independent:**
We have developed the package to be compatible with Linux and Windows operating systems.
This package has been developed Ubuntu 20.04.3 LTS, However, it has been tested on Windows 11 with python 3.10.14,Tensorflow 2.9.3 and Keras 2.9.0. 

## Using the pre-trained model to segment cytoplasmic vesicles
To create the cytomask, you need to place the cell_outline.mod file in the same directory as the tomogram.
You use the same script to build your pipeline, in case you are interested in all vesicles in tomograms you can set  in all vesicles within_segmentation_region = False.
We used object object-orientated approach to build the pipeline. You can use the following [script](notebooks/single_dataset_pathlib.py) to build your pipeline and run different steps of the pipeline.

You can use the following steps to run the pipeline on pre-trained model. 
The steps are briefly explained as follows:

0. Import the package
<pre>
import cryovesnet
</pre>
1. Set the directory of the tomogram 
<pre>
dataset_directory = Path("/mnt/data/tomogram_133/")
</pre>
2. Creating the pipeline  
<pre>
pl = cryovesnet.Pipeline(dataset_directory,pattern="*.rec.nad")
</pre>
> Here you can set the pattern to the file format of the tomogram in the directory. In case your tomogram is in the mrc format you can set the pattern to "*.mrc".
> The default form of Pipeline method is <pre> Pipeline(dataset_directory,pattern="*.rec.nad")</pre>

3. Setup the new directory  
<pre>
pl.setup_cryovesnet_dir()
</pre>
> You should have the cell_outline.mod file in the same directory as the tomogram. In case you do not have the cell_outline.mod file you can set make_mask = False.
By default, the setup_cryovesnet_dir function called like this: <pre>setup_cryovesnet_dir(make_masks=True, initialize=False, memkill=True) </pre>
4. Run the deep learning network
<pre>
pl.run_deep(force_run=True)
</pre>
> We are rescaling the mask to the original size of the tomogram.
You can set level of test time data augmentation by setting value of the augmentation_level. The default value is 1 which means no augmentation.
> The signature of the run_deep method is as follows:<pre> run_deep(force_run=False, rescale=None, gauss=True, augmentation_level=1, weight_path=None) </pre>

5. Rescale the mask to the original size of the tomogram
<pre>pl.rescale(force_run=True,slice_range=None)
</pre>
> You can set slice_range to clean the mask in a specific Z range, for example, to clean top and bottom of the mask you can set  you can set of the slice_range = (50,150)
> The signature of the rescale method is as follows:<pre> rescale(force_run=False, slice_range=None) </pre>
6. Generate primary labels if you do not have cell_outline.mod file in the same directory as the tomogram you can set within_segmentation_region = False
<pre>
pl.label_vesicles(within_segmentation_region = True,)
</pre>
> In case you have the cell_outline.mod file in the same directory as the tomogram you can set within_segmentation_region = True
> The global threshold is calculated to automatically. If you want to set the threshold on the mask you can set the threshold_coef to the value between 0 and 1.
> The definition of the label_vesicles method is as follows:
> <pre>label_vesicles(input_array_name='last_output_array_name', within_segmentation_region=False, threshold_coef=None,memkill=False)</pre>
7. Generate fine tuned labels
<pre>
pl.label_vesicles_adaptive(separating =True)
</pre>
> Adaptive thresholding is used to separate the vesicles closely packed together, and expanding the small vesicles.
> There are 3 main arguments in the label_vesicles_adaptive method, namely separating, expanding, and convex. The default values are False.
> "When not specified otherwise, label_vesicles_adaptive accepts these default parameters:
> <pre>label_vesicles_adaptive(expanding=False,convex=False, separating =False,  memkill=True)</pre>
8. Refinement using the radial profile
<pre>
pl.make_spheres()
</pre>
> The make_spheres method has the following signature:
> <pre>make_spheres(input_array_name='last_output_array_name', tight= False, keep_ellipsoid = False ,memkill=True) </pre>
> If you want to keep the ellipsoid shape of the vesicles you can set keep_ellipsoid = True.
9. Outlier detection and refinement
<pre>
pl.repair_spheres()
</pre>
> The repair_spheres method has the following signature:
> <pre>repair_spheres( p=0.3, m=4, memkill=True)/pre>
> which you can set the "m" as Mahalonobis distance threshold and p as p-value threshold, to remove outlier. 
10. This step ensures that the mod file is compatible with the pyto software
<pre>
pl.make_full_modfile(input_array_name='convex_labels')
pl.make_full_label_file()
</pre>

## Simple usage example
In the most simple case, you can use the following script to run the pipeline.
<pre>
import cryovesnet
pl = cryovesnet.run_default_pipeline('/mnt/data/amin/testtest/2')
</pre>
and if you want to see and edit the result you can use the following command.
<pre>
pl.fix_spheres_interactively()
</pre>

## Folder Structure

The ** indicates the files that are needed for the pipeline to run.
The single * indicates the files that are necessary for removing the outer membrane vesicles and creating the mod files.
After running the pipeline, "deep" folder and "cryovesnet" folder will be created.
The "deep" folder contains the output of the deep learning network and the "cryovesnet" folder contains the output of the post-processing steps.


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
└── <font color="#268BD2"><b>cryovesnet</b></font>
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
    ├── full_cryovesnet.mod
    ├── labels.mrc
    └── vesicles.mod

2 directories, 24 files</pre>


## Interactive cleaning of the segmentation
You can use the interactive cleaning of the segmentation to remove false positive and adding false negative vesicles.
You can use the following command to run the interactive cleaning of the sepecific label file.

<pre>
pl.fix_spheres_interactively("final_vesicle_labels")
</pre>

## Using Napari Tool without automation
You can use the following script to run the pipeline.
<pre>
import cryovesnet
dataset_directory = Path("/mnt/data/tomogram_133/")
pl = cryovesnet.Pipeline(dataset_directory,pattern="*.rec.nad")
pl.setup_cryovesnet_dir(make_masks= False, initialize=True)
pl.fix_spheres_interactively()
</pre>
## Windows users
> GPU support for native Windows ended with TensorFlow 2.10. From TensorFlow 2.11 onwards, you have the following options:
> 1. Install TensorFlow in Windows Subsystem for Linux 2 (WSL2)
> 2. Install the CPU-only version of TensorFlow, remove the tensorflow<2.10 from setup.py and install the package. We have tested with Keras < 3
> 3. Optionally, experiment with the TensorFlow-DirectML-Plugin
> 
> If you want to use GPU on native Windows, use this solution:
<pre>conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c conda-forge tensorflow=2.10 </pre>
and to verify the installation you can use the following command:
<pre>python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
</pre>


##  Runtime efficiency of the pipeline
The runtime of pipeline for a tomogram with pixel size 14.69, without making the mod file is around 200 second on a single GPU (~ A4000 Nvidia GPU).
Almost 20-30 pecent of the time is spent on running the pre-trained network. In the figure bellow, the runtime of the pipeline is shown. (with mod file generation)
![Pipeline](images/efficiency.png)
We have tested the pipeline on a Macbook Pro M1 without any GPU utilization, and whole procedure took around 27.5 minutes for a tomogram with size 1024x1024x261 voxels and voxel size of 14.69.
## Train and create your own dataset
The notebooks allow to [generate the training data](notebooks/create_trainingset.ipynb) and [train the network](notebooks/training_vesicles.ipynb). Pre-trained weights for the network are provided in the weights folder.

## Pyto

This repository includes scripts or code from external packages. To ensure integration or extended capabilities, inclusions is from the [Pyto](https://github.com/vladanl/Pyto) project. In case you want to segment the connectors and tethers afterwards you can use this software.

## Citation

If you utilize our software in your research, please acknowledge its use by citing this work. Our software may integrate or rely on various third-party tools; hence, we recommend citing those as appropriately detailed in their documentation.
https://doi.org/10.1101/2024.02.26.582080

<details>
<summary>Bibtex</summary>
<p>

```bibtex
@article{Khosrozadeh2024.02.26.582080,
    author = {Amin Khosrozadeh and Raphaela Seeger and Guillaume Witz and Julika Radecke and Jakob B. S{\o}rensen and Beno{\^\i}t Zuber},
    title = {CryoVesNet: A Dedicated Framework for Synaptic Vesicle Segmentation in Cryo Electron Tomograms},
    elocation-id = {2024.02.26.582080},
    year = {2024},
    doi = {10.1101/2024.02.26.582080},
    URL = {https://www.biorxiv.org/content/early/2024/02/28/2024.02.26.582080},
    eprint = {https://www.biorxiv.org/content/early/2024/02/28/2024.02.26.582080.full.pdf},
    journal = {bioRxiv}
}
```
</p>
</details>


## Schematic Workflow
![Pipeline](images/github_figure.png)
