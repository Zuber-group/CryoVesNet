from setuptools import setup

setup(
    name='prepyto',
    version='0.1.1',
    url='',
    packages=[".", 'unetmic', 'unetmic.unetmic', 'weights','pyto_scripts',],
    license='',
    author='Amin Khosrozadeh',
    author_email='',
    description='prepyto - branch pathlib',
    install_requires=['numpy', 'scikit-image', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari'],
)
