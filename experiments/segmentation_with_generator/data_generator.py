import os
import cv2
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    def __init__(self, images_dir, masks_dir, batch_size=32, dim=(256, 256), n_channels=1,
                 n_classes=1, shuffle=True, preprocessor=None, augmentations=None):
        # Store image and mask directories
        self.img_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)

        # Only use images which have a corresponding mask
        self.ids = set(self.img_ids).intersection(set(self.mask_ids))

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Store image attributes
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # End-of-epoch behaviour
        self.indexes = np.arange(len(self.ids))
        self.shuffle = shuffle
        self.on_epoch_end()

        # Preprocessor and data augmentations
        self.preprocessor = preprocessor
        self.augmentations = augmentations

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of dirs
        image_mask_dirs = [(self.images_fps[k], self.masks_fps[k]) for k in indexes]

        # Generate data
        return self.__data_generation(image_mask_dirs)

    def on_epoch_end(self):
        """ Updates indexes after each epoch and stores the model performance """
        self.indexes = np.arange(len(self.ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_mask_dirs):
        """ Generates data containing batch_size samples """  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, dirs in enumerate(image_mask_dirs):
            # Store image
            x_img = cv2.imread(dirs[0])
            X[i,] = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

            # Store mask
            y_img = cv2.imread(dirs[1], cv2.IMREAD_GRAYSCALE).reshape((*self.dim, 1))
            y[i,] = y_img

        if self.preprocessor is not None:
            X = self.preprocessor(X)
            y = self.preprocessor(y)

        X = X.astype('float32')
        X /= 255
        y = y.astype('float32')
        y /= 255

        return X, y
