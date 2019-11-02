from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
import numpy as np
import cv2
import argparse
import time

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to saved model")

args = parser.parse_args()

# Load a pre-trained segmentation model
model = load_model(args.model, custom_objects={"binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                                             "iou_score": sm.metrics.iou_score})

# The desired image dimensions and camera frame rate
dim = (256, 256)
frame_rate = 10  # fps

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dim[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dim[1])


# Get an estimate of the camera frame rate to be able to adjust the demo's frame rate
def get_frame_rate(video):
    # Number of frames to capture
    num_frames = 30

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        _, _ = video.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start

    # Calculate (rounded) frames per second
    return num_frames / seconds

print("Getting frame rate...")
curr_rate = get_frame_rate(cap)
thresh = int(curr_rate // frame_rate)

print("Starting camera...")
frames_since_reset = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frames_since_reset % thresh == 0:
        frames_since_reset = 1
        # Format the frame and predict the mask
        try:
            frame = np.array(cv2.resize(frame, dim), dtype='float32')
            input_frame = frame.reshape((1, *dim, 3))
            input_frame /= 255
            mask = model.predict(input_frame)
            mask = mask.reshape((*dim, 1))

            # Change the mask data format
            mask *= 255
            mask = mask.astype(np.uint8)
            np_mask = mask > 128

            # Display the resulting frame (with superimposed mask)
            result = input_frame.copy()[0, :, :, :]
            result[np_mask[:, :, 0], 1] = 0

            cv2.imshow('demo', cv2.flip(result, 1))
        except Exception as e:
            print("Skipping frame; Exception occurred: '{}'".format(e))

    frames_since_reset += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print("Stopping camera...")
cap.release()
cv2.destroyAllWindows()
