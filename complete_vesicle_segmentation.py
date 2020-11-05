import os
import unetmic.segment as segment
from pathlib import Path
import tensorflow as tf
import shutil

def vesicle_segmentation(path_to_folder):
    os.chdir('../..')
    dataset_dir = os.getcwd()
    unet_weigth_path = dataset_dir + '/weights/weights.h5'
    os.chdir('./data/'+path_to_folder)

    if os.path.exists('./' + '/deep'):
        shutil.rmtree('deep', ignore_errors=True)
        os.mkdir('deep')
    else:
        os.mkdir('deep')
    cwd = Path('.')
    path_generator = cwd.glob('*.rec')
    file_name = ([str(x) for x in path_generator])[0]
    print(file_name)
    path_to_file=file_name

    # if 'session' in locals() and session is not None:
    #     print('Close interactive session')
    #     session.close()

    if tf.__version__[0] == '1':
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
        print('Check if GPU is available')
        print(tf.test.gpu_device_name())
    else:
        print('Check if GPU is available')
        print(tf.config.list_physical_devices())
    # set path to unet weights

    folder_to_save = 'deep/'

    # set network size
    network_size = 64
    segment.full_segmentation(network_size, unet_weigth_path, path_to_file, folder_to_save, rescale=0.5, gauss=True)
