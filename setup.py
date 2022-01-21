from setuptools import setup, find_packages

setup(
    name='prepyto',
    version='0.1.1',
    url='',
    packages=find_packages(),
    license='',
    author='Amin Khosrozadeh',
    author_email='',
    description='prepyto - branch pathlib',
    package_data={'': ['weights/weights.h5']},
    include_package_data=True,
    install_requires=['numpy', 'scikit-image', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari',
                      'keras'],

)
