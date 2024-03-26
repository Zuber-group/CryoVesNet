from setuptools import setup, find_packages

setup(
    name='cryovesnet',
    version='0.2.0',
    url='',
    packages=find_packages(),
    license='',
    author='Amin Khosrozadeh',
    author_email='',
    description='cryovesnet - branch pathlib',
    package_data={'': ['weights/weights.h5']},
    include_package_data=True,
    install_requires=['numpy<1.24', 'scikit-image<0.19', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari','pydantic<2',
                      'napari-mrcfile-reader','keras','tensorflow<2.10'],

)
