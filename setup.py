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

