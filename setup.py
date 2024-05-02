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
    install_requires=['numpy', 'scikit-image', 'scipy', 'jupyter','jupyterlab',
                      'pandas', 'h5py', 'tifffile','mrcfile','tqdm', 'napari[pyqt5]','pydantic<2',
                      'napari-mrcfile-reader','keras==2.9.0','tensorflow<2.10'],

)
