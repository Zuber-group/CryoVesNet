import os
import datetime
import json
import numpy as np
from tqdm import tqdm, trange
import skimage.io
import mrcfile
import tensorflow as tf

try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.layers import Input, Conv3D, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation, Add, \
        Conv2DTranspose
except ModuleNotFoundError:
    print("unetmic: tensorflow is missing, some function will fail")

# from keras import backend as K
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Input, Conv3D

from . import blocks


def train_generator(folder, numtot, batchsize):
    rand_arr = np.random.choice(numtot, size=numtot, replace=False)
    num = 0
    while num > -1:

        batch_counter = 0

        im_batch = []
        mask_batch = []
        for i in range(batchsize):
            im_batch.append(np.load(folder + '/image_' + str(rand_arr[num]) + '.npy')[:, :, :, np.newaxis])
            mask_batch.append(np.load(folder + '/mask_' + str(rand_arr[num]) + '.npy')[:, :, :, np.newaxis])
            num += 1
            if num == numtot:
                num = 0
                rand_arr = np.random.choice(numtot, size=numtot, replace=False)

        im_batch = (im_batch - np.mean(im_batch)) / np.std(im_batch)
        yield (np.array(im_batch).astype(np.float32), np.array(mask_batch).astype(np.float32))


def generator_2d(folder, subset_size, batch_size, start_index=0):
    permutation = np.random.permutation(subset_size) if start_index == 0 else np.arange(subset_size)
    num = 0
    while num > -1:
        im_batch = []
        mask_batch = []
        for _ in range(batch_size):
            slice_id = permutation[num] + start_index
            im_batch.append(np.load(folder + f'/image_{slice_id}.npy')[..., np.newaxis])
            mask_batch.append(np.load(folder + f'/mask_{slice_id}.npy')[..., np.newaxis])
            num += 1
            if num == subset_size:
                num = 0
                permutation = np.random.permutation(subset_size)
        im_batch = np.array(im_batch).astype(np.float32)
        mask_batch = np.array(mask_batch).astype(np.float32)
        im_batch = (im_batch - np.mean(im_batch)) / np.std(im_batch)
        yield im_batch, mask_batch


def valid_generator(folder, numvalid, batchsize, numtot):
    num = 0
    while num > -1:
        im_batch = []
        mask_batch = []
        for i in range(batchsize):
            im_batch.append(np.load(folder + '/image_' + str(num + numtot) + '.npy')[:, :, :, np.newaxis])
            mask_batch.append(np.load(folder + '/mask_' + str(num + numtot) + '.npy')[:, :, :, np.newaxis])
            num += 1
            if num == numvalid:
                num = 0

        im_batch = (im_batch - np.mean(im_batch)) / np.std(im_batch)
        yield (np.array(im_batch).astype(np.float32), np.array(mask_batch).astype(np.float32))


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def weighted_binary_crossentropy(y_true, y_pred):
    # Assign weights
    class_weight_0 = 1.0
    class_weight_1 = 10.0

    # Calculate the binary cross-entropy loss
    bce = K.binary_crossentropy(y_true, y_pred)

    # Apply the weights
    weight_vector = y_true * class_weight_1 + (1. - y_true) * class_weight_0
    weighted_bce = weight_vector * bce

    # Return the mean error
    return K.mean(weighted_bce)


def create_resnet(inputs):
    # Residual block 1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    shortcut = Conv2D(32, (1, 1), padding='same')(inputs)
    add1 = Add()([shortcut, conv1])
    act1 = Activation('relu')(add1)
    # Residual block 2I
    conv2 = Conv2D(64, (3, 3), padding='same')(act1)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    shortcut2 = Conv2D(64, (1, 1), padding='same')(act1)
    add2 = Add()([shortcut2, conv2])
    act2 = Activation('relu')(add2)
    # Residual block 3
    conv3 = Conv2D(128, (3, 3), padding='same')(act2)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    shortcut3 = Conv2D(128, (1, 1), padding='same')(act2)
    add3 = Add()([shortcut3, conv3])
    act3 = Activation('relu')(add3)
    # Residual block 4
    conv4 = Conv2D(256, (3, 3), padding='same')(act3)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    shortcut4 = Conv2D(256, (1, 1), padding='same')(act3)
    add4 = Add()([shortcut4, conv4])
    act4 = Activation('relu')(add4)
    output = Conv2D(1, (1, 1), activation='sigmoid')(act4)
    return output


def inception_module(inputs, filters=64):
    t1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)

    t2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    t2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(t2)

    t3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    t3 = Conv2D(filters, (5, 5), padding='same', activation='relu')(t3)

    t4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    t4 = Conv2D(filters, (1, 1), padding='same', activation='relu')(t4)

    return Concatenate()([t1, t2, t3, t4])


def create_unet_deep_2d(inputs):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    up4 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    return Conv2D(1, (1, 1), activation='sigmoid')(conv5)


def create_eman2(inputs):
    conv1 = Conv2D(40, (15, 15), activation='relu', padding='same')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(40, (15, 15), activation='relu', padding='same')(max1)
    conv3 = Conv2D(1, (15, 15), activation='sigmoid', padding='same')(conv2)
    return UpSampling2D(size=(2, 2))(conv3)


def create_inception(inputs):
    inception1 = inception_module(inputs)
    inception2 = inception_module(inception1)
    inception3 = inception_module(inception2)
    return Conv2D(1, (1, 1), padding='same', activation="sigmoid")(inception3)


def create_vgg(inputs):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 2
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block 3
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Upsampling and Decoding
    up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(pool3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

    up3 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(conv8)
    return Conv2D(1, (1, 1), activation='sigmoid')(up3)

# 2D networks adopted from https://github.com/bionanopatterning/Ais/tree/master/Ais/models
def create_unet_2d(inputsize=128, network_name='eman2'):
    inputs = Input((inputsize, inputsize, 1))
    net_factory = {
        'eman2': create_eman2,
        'inception': create_inception,
        'resnet': create_resnet,
        'unet': create_unet_deep_2d,
        'vgg': create_vgg
    }[network_name]
    model = Model(inputs=inputs, outputs=net_factory(inputs))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model


def create_unet_3d(inputsize=(32, 32, 32, 1),
                   n_depth=2,
                   n_filter_base=16,
                   kernel_size=(3, 3, 3),
                   activation='relu',
                   batch_norm=True,
                   dropout=0.0,
                   n_conv_per_depth=2,
                   pool_size=(2, 2, 2),
                   n_channel_out=1):
    n_dim = len(kernel_size)

    input = Input(inputsize, name="input")

    unet = blocks.unet_block(n_depth, n_filter_base, kernel_size,
                             activation=activation, dropout=dropout, batch_norm=batch_norm,
                             n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = Conv3D(n_channel_out, (1,) * n_dim, activation='sigmoid')(unet)

    unet3d = Model(inputs=input, outputs=final)
    unet3d.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    # For multi-gpu training you need to use the following line instead of the previous one
    # unet3d.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=[dice_coef])
    return unet3d


def run_training(network, save_folder, folder, numtot, batchsize, numvalid, is_3d=True):
    with open(save_folder + 'arguments.json', 'w') as f:
        json.dump({'save_folder': save_folder, 'folder': folder, 'numtot': numtot, 'batchsize': batchsize,
                   'numvalid': numvalid}, f)

    model_checkpoint = ModelCheckpoint(save_folder + 'weights.h5', monitor='val_loss', save_best_only=True)

    # model_checkpoint = ModelCheckpoint(save_folder + 'weights.h5', monitor='val_dice_coef', mode='max',save_best_only=True)
    # earlystopping = EarlyStopping(
    #     monitor="val_dice_coef",
    #     min_delta=0,
    #     patience=10,
    #     verbose=1,
    #     mode="max",
    #     baseline=None,
    #     restore_best_weights=True,
    # )
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logdir, 'training.log'))
    # network.fit_generator(generator=train_generator(folder, numtot, batchsize),
    #                       validation_data=valid_generator(folder, numvalid, numtot, batchsize),
    #                       validation_steps=np.floor(numvalid / batchsize),
    #                       steps_per_epoch=np.floor(numtot / batchsize),
    #                       epochs=200, verbose=1, callbacks=[model_checkpoint,earlystopping], class_weight=[1, 10])

    network.fit_generator(generator=(train_generator if is_3d else generator_2d)(folder, numtot, batchsize),
                          validation_data=(valid_generator if is_3d else generator_2d)(folder, numvalid, batchsize,
                                                                                       numtot),
                          validation_steps=np.floor(numvalid / batchsize),
                          steps_per_epoch=np.floor(numtot / batchsize),
                          epochs=200 if is_3d else 1000, verbose=1,
                          callbacks=[model_checkpoint, tensorboard_callback, csv_logger],
                          class_weight=[1, 10])

    tf.keras.backend.clear_session()


# def run_training_2d(network, save_folder, data_folder, numtot, batchsize, numvalid):
#     # TODO: for 2d: generator=generator_2d(folder, numtot, batchsize), validation_data=generator_2d(folder, numvalid, batchsize, start_index=numtot)
#     # save all arguments in a json file
#     with open(save_folder + 'arguments.json', 'w') as f:
#         json.dump({'save_folder': save_folder, 'data_folder': data_folder, 'numtot': numtot, 'batchsize': batchsize,
#                    'numvalid': numvalid}, f)
#
#     model_checkpoint = ModelCheckpoint(save_folder + 'weights.h5', monitor='val_loss', save_best_only=True)
#     network.fit_generator(generator=generator_2d(data_folder, numtot, batchsize),
#                           validation_data=generator_2d(data_folder, numvalid, batchsize, start_index=numtot),
#                           validation_steps=np.floor(numvalid / batchsize),
#                           steps_per_epoch=np.floor(numtot / batchsize),
#                           epochs=1000, verbose=1, callbacks=[model_checkpoint], class_weight=[1, 10],
#                           use_multiprocessing=False, workers=1)


def run_training_multiGPU(save_folder, data_folder, num_total, batch_size, num_valid, window_size=64):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    data_name = data_folder.split('/')[-1]
    if data_name == '':
        data_name = data_folder.split('/')[-2]
    logdir = os.path.join(save_folder, f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{data_name}_{window_size}')
    os.makedirs(logdir, exist_ok=True)
    with open(logdir + '/arguments.json', 'w') as f:
        json.dump({'save_folder': save_folder, 'folder': data_folder, 'numtot': num_total,
                   'batchsize': batch_size, 'numvalid': num_valid}, f)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logdir, 'log.csv'), append=True, separator=';')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    with strategy.scope():
        # Create and compile the model within the strategy scope
        network = create_unet_3d(inputsize=(window_size, window_size, window_size, 1),
                                 n_depth=2 if window_size == 64 else 2,
                                 n_filter_base=window_size // 4,
                                 kernel_size=(3, 3, 3),
                                 activation='relu',
                                 batch_norm=True,
                                 dropout=0.0,
                                 n_conv_per_depth=2,
                                 pool_size=(2, 2, 2),
                                 n_channel_out=1)
        network.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=[dice_coef])

        # Setup callbacks
        model_checkpoint_dice = ModelCheckpoint(
            logdir + '/weights_best_dice.h5',
            monitor='val_dice_coef', mode='max', save_best_only=True)
        model_checkpoint_loss = ModelCheckpoint(
            logdir + '/weights_best_loss.h5',
            monitor='val_loss', mode='min', save_best_only=True)
        # earlystopping = EarlyStopping(monitor="val_dice_coef", min_delta=0, patience=50, verbose=1, mode="max",
        #                               restore_best_weights=True)

        # Convert generators to tf.data.Dataset
        train_data = tf.data.Dataset.from_generator(lambda: train_generator(data_folder, num_total, batch_size),
                                                    output_types=(tf.float32, tf.float32),
                                                    output_shapes=(
                                                        [batch_size, window_size, window_size, window_size, 1],
                                                        [batch_size, window_size, window_size, window_size, 1]))
        valid_data = tf.data.Dataset.from_generator(
            lambda: valid_generator(data_folder, num_valid, batch_size, num_total),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [batch_size, window_size, window_size, window_size, 1],
                [batch_size, window_size, window_size, window_size, 1]))

        # Train the model
        network.fit(train_data, epochs=200, steps_per_epoch=np.floor(num_total / batch_size),
                    validation_data=valid_data, validation_steps=np.floor(num_valid / batch_size),
                    callbacks=[model_checkpoint_dice, model_checkpoint_loss, csv_logger, tensorboard_callback])

    tf.keras.backend.clear_session()


def run_training_multiGPU_2d(save_folder, data_folder, num_total, batch_size, num_valid, window_size=128,
                             network_name='eman2'):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    data_name = data_folder.split('/')[-1]
    if data_name == '':
        data_name = data_folder.split('/')[-2]
    logdir = os.path.join(save_folder,
                          f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{data_name}_{network_name}')
    os.makedirs(logdir, exist_ok=True)
    with open(logdir + '/arguments.json', 'w') as f:
        json.dump({'save_folder': save_folder, 'folder': data_folder, 'numtot': num_total,
                   'batchsize': batch_size, 'numvalid': num_valid}, f)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logdir, 'log.csv'), append=True, separator=';')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    with strategy.scope():
        # Create and compile the model within the strategy scope
        network = create_unet_2d(inputsize=window_size, network_name=network_name)
        network.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=[dice_coef])

        # Setup callbacks
        model_checkpoint_dice = ModelCheckpoint(
            logdir + '/weights_best_dice.h5',
            monitor='val_dice_coef', mode='max', save_best_only=True)
        model_checkpoint_loss = ModelCheckpoint(
            logdir + '/weights_best_loss.h5',
            monitor='val_loss', mode='min', save_best_only=True)
        # earlystopping = EarlyStopping(monitor="val_dice_coef", min_delta=0, patience=50, verbose=1, mode="max",
        #                               restore_best_weights=True)

        # Convert generators to tf.data.Dataset
        train_data = tf.data.Dataset.from_generator(lambda: generator_2d(data_folder, num_total, batch_size),
                                                    output_types=(tf.float32, tf.float32),
                                                    output_shapes=(
                                                        [batch_size, window_size, window_size, 1],
                                                        [batch_size, window_size, window_size, 1]))
        valid_data = tf.data.Dataset.from_generator(
            lambda: generator_2d(data_folder, num_valid, batch_size, start_index=num_total),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [batch_size, window_size, window_size, 1],
                [batch_size, window_size, window_size, 1]))

        # Train the model
        network.fit(train_data, epochs=1000, steps_per_epoch=np.floor(num_total / batch_size),
                    validation_data=valid_data, validation_steps=np.floor(num_valid / batch_size),
                    callbacks=[model_checkpoint_dice, model_checkpoint_loss, csv_logger, tensorboard_callback])

    tf.keras.backend.clear_session()


def run_segmentation(image, unet3d,roi=24):
    # region of interest. Middle image region kept after segmentation
    # the network input needs to be larger than this (e.g. 64) to avoid
    # tiling effects

    network_size = unet3d.input.get_shape().as_list()[1]
    # roi = 24
    print("ROI: ", roi)
    wsize = network_size
    to_pad0 = roi - (np.array(image.shape) % roi)
    to_pad = int((wsize - roi) / 2)

    # pad image to avoid edge effects
    image_pd = np.pad(image, [[0, x] for x in roi - (np.array(image.shape) % roi)], mode='constant', constant_values=0)
    ideal_size = np.array(image_pd.shape)
    image_pd = np.pad(image_pd, to_pad, mode='constant', constant_values=0)

    # run the segmentation by tiling the image
    image_mask = np.zeros(ideal_size)
    numit = 0
    for i in tqdm(range(0, image.shape[0], roi)):
        # print(i)
        for j in range(0, image.shape[1], roi):
            # print(j)
            for k in range(0, image.shape[2], roi):
                # print(k)
                numit += 1
                test = unet3d.predict(image_pd[i:i + wsize, j:j + wsize, k:k + wsize][np.newaxis, :, :, :, np.newaxis])
                test_im = test[0, :, :, :, 0]
                image_mask[i:i + roi, j:j + roi, k:k + roi] \
                    = test_im[to_pad:to_pad + roi, to_pad:to_pad + roi, to_pad:to_pad + roi]
    image_mask = image_mask[0:image_mask.shape[0] - to_pad0[0], 0:image_mask.shape[1] - to_pad0[1],
                 0:image_mask.shape[2] - to_pad0[2]]
    K.clear_session()
    return image_mask


def gaussian_kernel_3d(kernel_size, sigma=1.0):
    offset = (kernel_size - 1) / 2
    ax = np.linspace(-offset, offset, kernel_size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def run_segmentation_overlap(image, unet3d, roi=24, use_kernel=True, batch_size=1):
    # region of interest. Middle image region kept after segmentation
    # the network input needs to be larger than this (e.g. 64) to avoid
    # tiling effects
    network_size = unet3d.input.get_shape().as_list()[1]
    wsize = network_size
    to_pad = int((wsize - roi) / 2)
    input_shape = image.shape

    # pad image to avoid edge effects
    image_pd = np.pad(image, [[0, x] for x in roi - (np.array(image.shape) % roi)], mode='constant', constant_values=0)
    image_pd = np.pad(image_pd, to_pad, mode='constant', constant_values=0)

    # run the segmentation by tiling the image
    mask_pd = np.zeros_like(image_pd)
    segmentation_weight = np.zeros_like(image_pd)
    kernel = gaussian_kernel_3d(wsize) if use_kernel else 1
    # current_batch = []
    # i_list = []
    # j_list = []
    # k_list = []
    for i in trange(0, image.shape[0], roi):
        for j in trange(0, image.shape[1], roi, leave=False):
            for k in trange(0, image.shape[2], roi, leave=False):
                test_im = unet3d.predict(image_pd[i:i + wsize, j:j + wsize, k:k + wsize][np.newaxis, :, :, :,
                                         np.newaxis])[0, ..., 0]
                mask_pd[i:i + wsize, j:j + wsize, k:k + wsize] += test_im * kernel
                segmentation_weight[i:i + wsize, j:j + wsize, k:k + wsize] += kernel
                # current_batch.append(image_pd[i:i + wsize, j:j + wsize, k:k + wsize])
                # i_list.append(i)
                # j_list.append(j)
                # k_list.append(k)
                # if len(current_batch) == batch_size or (
                #         i + roi >= image.shape[0] and j + roi >= image.shape[1] and k + roi >= image.shape[2]):
                #     test_im = unet3d.predict(np.stack(current_batch, axis=0)[..., np.newaxis])[..., 0]
                #     for c, (ii, jj, kk) in enumerate(zip(i_list, j_list, k_list)):
                #         mask_pd[ii:ii + wsize, jj:jj + wsize, kk:kk + wsize] += test_im[c] * kernel
                #         segmentation_weight[ii:ii + wsize, jj:jj + wsize, kk:kk + wsize] += kernel
                #     current_batch = []
                #     i_list = []
                #     j_list = []
                #     k_list = []
    mask_pd = mask_pd[to_pad:to_pad + input_shape[0], to_pad:to_pad + input_shape[1], to_pad:to_pad + input_shape[2]]
    segmentation_weight = segmentation_weight[to_pad:to_pad + input_shape[0], to_pad:to_pad + input_shape[1],
                          to_pad:to_pad + input_shape[2]]
    return mask_pd / segmentation_weight


def run_segmentation_2d(image, unet_2d, roi=64):
    # region of interest. Middle image region kept after segmentation
    # the network input needs to be larger than this (e.g., network size) to avoid tiling effects
    network_size = unet_2d.input.get_shape().as_list()[1]
    wsize = network_size
    to_pad0 = roi - (np.array(image.shape[1:]) % roi)
    to_pad = int((wsize - roi) / 2)

    # pad image to avoid edge effects
    padding = [(0, 0)]  # no padding for the first dimension (slices)
    padding += [(to_pad, to_pad + to_pad0[i]) for i in range(2)]  # padding for height and width
    image_pd = np.pad(image, padding, mode='constant', constant_values=0)

    # Initialize mask with zeros
    ideal_size = np.array(image_pd.shape)
    image_mask = np.zeros(ideal_size)

    # run the segmentation by iterating over each slice
    for i in tqdm(range(image.shape[0])):  # iterating over slices
        for j in range(0, image.shape[1], roi):
            for k in range(0, image.shape[2], roi):
                # Ensure not to exceed image dimensions
                slice_pd = image_pd[i, j:j + wsize, k:k + wsize]
                test = unet_2d.predict(slice_pd[np.newaxis, :, :, np.newaxis])
                test_im = test[0, :, :, 0]

                # Update the corresponding region in the mask
                image_mask[i, j + to_pad:j + to_pad + roi, k + to_pad:k + to_pad + roi] = \
                    test_im[to_pad:to_pad + roi, to_pad:to_pad + roi]

    # Remove padding from the mask to match original image dimensions
    image_mask = image_mask[:, to_pad:image.shape[1] + to_pad, to_pad:image.shape[2] + to_pad]
    return image_mask


def load_raw(path_to_file):
    if os.path.splitext(os.path.split(path_to_file)[1])[1] == '.tif':
        image = skimage.io.imread(path_to_file)

    else:
        image = mrcfile.open(path_to_file)
        image = image.data

    return image


def save_seg_output(image_mask, path_to_file, folder_to_save):
    # check if folder exists
    if not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)

    # save the resulting mask as npy file
    np.save(
        os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + '_segUnet.npy',
        image_mask)
    # create an 8bit image that one can visualize in Fiji and save it
    # image_mask_int = (255*np.flip(image_mask,axis = 1)).astype(np.uint8)
    image_mask_int = (255 * image_mask).astype(np.uint8)
    skimage.io.imsave(
        os.path.normpath(folder_to_save) + '/' + os.path.splitext(os.path.split(path_to_file)[1])[0] + '_segUnet.tiff',
        image_mask_int, plugin='tifffile')
