# https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/
# https://medium.com/@sathualab/how-to-use-tensorflow-graph-with-opencv-dnn-module-3bbeeb4920c5
# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

# https://www.youtube.com/watch?v=ihQdg0kiLR4

"""
python3 segment_video.py --model enet-cityscapes/enet-model.net \
    --classes enet-cityscapes/enet-classes.txt \
    --colors enet-cityscapes/enet-colors.txt \
    --video videos/massachusets.mp4 \
    --output output/output_massachusets.mp4

"""

import numpy as np
import argparse
import imutils
import time
import cv2

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", required=True, help="path to deep learning segmentation model"
)
ap.add_argument(
    "-c", "--classes", required=True, help="path to .txt file containing class labels"
)
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", required=True, help="path to output video file")
ap.add_argument(
    "-s",
    "--show",
    type=int,
    default=1,
    help="whether or not to display frame to screen",
)
ap.add_argument(
    "-l", "--colors", type=str, help="path to .txt file containing colors for labels"
)
ap.add_argument(
    "-w",
    "--width",
    type=int,
    default=500,
    help="desired width (in pixels) of input image",
)
args = vars(ap.parse_args())

# Load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# If a colors file was supplied, load it from disk
if args["colors"]:
    COLORS = open(args["colors"]).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

# Otherwise we need to randomly generate RGB colors for each class label
else:
    # Initialize a list of colors to represent each class label in
    # the mask (strating with 'black' for the background /unlabelled
    # regions)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3), dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# Initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

# Loop over the class name + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
    # Draw the class name + color on the legend
    color = [int(c) for c in color]
    cv2.putText(
        legend,
        className,
        (5, (i * 25) + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# Initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["video"])
writer = None

# Try to determine the total number of frames in the video file
try:
    prop = (
        cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    )
    total = int(vs.get(prop))
    print("[INFO] Total frames in video".format(total))

# An error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] Could not determine # of frames in video")
    total = -1

# Loop over the frames from the video file stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached
    # the end of the stream
    if not grabbed:
        break

    # Construct a blob from the frame and perform a forward
    # pass using the segmentation model
    frame = imutils.resize(frame, width=args["width"])
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False
    )
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    # Infer the total number of classes along with the spatial dimensions
    # of the mask image via the shape of the output array
    (numClasses, height, width) = output.shape[1:4]

    # Output class ID map will be num_classes x height y width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-cooridnate in the image
    classMap = np.argmax(output[0], axis=0)

    # Given the class ID map, we can map each of the class IDs to its
    # corresponding color
    mask = COLORS[classMap]

    # Resize the mask and class map such that its dimensions match the
    # original size of the input frame
    mask = cv2.resize(
        mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Perform a weighted combination of the input image with the mask to
    # form an output visualization
    output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")

    # Check the video writer
    if writer is None:
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, 30, (output.shape[1], output.shape[0]), True
        )

        # Information on processing single frame
        if total > 0:
            elap = end - start
            print("[INFO] single frame took {:4f} seconds".format(elap))
            print("[INFO] estimated total time {:4f}".format(total * elap))

        # Write the output frame to disk
        writer.write(output)

    # Display the frame on the screen
    if args["show"] > 0:
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed break the loop
        if key == ord("q"):
            break

# Release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
