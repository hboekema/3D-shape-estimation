from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt
import segmentation_models as sm
import argparse
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to the pre-trained model to be used for predicting")
parser.add_argument("images", help="Path to the directory containing the images to predict on")

args = parser.parse_args()

# Load a pre-trained segmentation model
model = load_model(args.model, custom_objects={"binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                                             "iou_score": sm.metrics.iou_score})

generator_params = {
    "rescale": 1./255,
    "shear_range": 0.2,
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "dtype": "float32"
}

image_datagen = ImageDataGenerator(**generator_params)

image_generator = image_datagen.flow_from_directory(
        args.images,
        target_size=(256, 256),
        batch_size=32,
        color_mode="rgb",
        class_mode=None,
)

preds = model.predict_generator(
    image_generator,
    steps=1,
    use_multiprocessing=True
)

preds = preds.reshape(preds.shape[0], 256, 256)
preds *= 255
preds.astype(np.uint8)

for pred in preds:
    plt.imshow(pred, cmap='gray')
    plt.show()
