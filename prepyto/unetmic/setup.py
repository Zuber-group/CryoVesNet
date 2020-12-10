from setuptools import setup

setup(name='unetmic',
      version='0.1',
      description='Unet for microscopy segmentation at MIC',
      url='https://github.com/guiwitz',
      author='Guillaume Witz',
      author_email='',
      license='MIT',
      packages=['unetmic'],
      zip_safe=False,
      install_requires=['numpy==1.16.4','scikit-image==0.16.1','scipy','jupyter','jupyterlab','pandas','h5py','tifffile','mrcfile','tqdm','napari'],#,
                       #'tensorflow-gpu==1.14.0', 'keras==2.2.4'],
      )
