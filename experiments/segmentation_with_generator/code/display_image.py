import argparse
import matplotlib.pyplot as plt
import cv2

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to image to be displayed")
parser.add_argument("--mask", "-m", action="store_true", help="Flag indicating whether this image is a mask")

args = parser.parse_args()

# Load the image using cv2
if args.mask:
	img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
else:
	img = cv2.CvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

# Process the image such that it is in the right format
#img /= 255
#img = img.astype('float32')

# Display the image using matplotlib
if args.mask:
	plt.imshow(img, cmap='gray')
else:
	plt.imshow(img)
plt.show()
