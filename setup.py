from setuptools import setup, find_packages

setup(
    name='CryoVesNet',
    version='0.5.0',
    url='https://github.com/Zuber-group/CryoVesNet',
    packages=find_packages(),
    license='MIT',  # Please confirm or change the license as appropriate
    author='Amin Khosrozadeh, Guillaume Witz, Benoît Zuber',
    author_email='amin.khosrozadeh@unibe.ch',
    description='CryoVesNet: A Dedicated Framework for Synaptic Vesicle Segmentation in Cryo Electron Tomograms',
    package_data={'': ['weights/weights.h5']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy',
        'jupyter',
        'jupyterlab',
        'pandas',
        'h5py',
        'tifffile',
        'mrcfile',
        'tqdm',
        'napari[pyqt5]',
        'napari-mrcfile-reader',
        'keras==2.9.0',
        'tensorflow<2.10'
    ],
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)