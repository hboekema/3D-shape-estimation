from keras.models import load_model
import segmentation_models as sm
import numpy as np
import cv2

# Load a pre-trained segmentation model
model_fp = "../models/model-2019-10-17 16:24:52.h5"
model = load_model(model_fp, custom_objects={"binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                                             "iou_score": sm.metrics.iou_score})

# The desired image dimensions
dim = (256, 256)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dim[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dim[1])

def overlay_mask(img, mask):
    fg = cv2.bitwise_or(img, img, mask=mask)

    return fg

print("Starting camera...")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.resize(frame, dim)
    input_frame = frame.reshape((1, *dim, 3))
    mask = model.predict(input_frame)
    mask = mask.reshape((*dim, 1))

    # Change the mask data format
    mask *= 255
    mask = mask.astype(np.uint8)

    # Display the resulting frame
    result = overlay_mask(frame, mask)
    cv2.imshow('mask', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

