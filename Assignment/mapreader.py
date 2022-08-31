#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.

import statistics
import sys, cv2, numpy, math

#-------------------------------------------------------------------------------

# Exposition

# This program is used to extract the bearing and position of a triangular pointer
# within an image of a map
# The fist part deals with defining cropping the original image to a more manageable
# and correct size, removing unnecessary edges using contours.
# After that, using the inRange function, the green arrow is identified, and based
# on its relative position, the image is modified to face the correct way.
# The final part again uses color identification and contours to identify the red
# triangle and segment it from the image. Once this triangle was found, using the
# minEnclosingTriangle function, the points were found and finding the distance
# between each point in order to determine the direction it was facing. Finally
# atan2 is used to get the angle of the triangle and some normalization is needed
# to adapt the values to the required scale.

#-------------------------------------------------------------------------------

# Functions

# Function to display images used in debugging

def dispRes(resIm, name):
    cv2.namedWindow (name, cv2.WINDOW_NORMAL)
    if len(resIm.shape) == 2:
        ny, nx = resIm.shape
    else:
        ny, nx, nc = resIm.shape
    cv2.resizeWindow (name, nx//2, ny//2)
    cv2.imshow (name, resIm)
    cv2.waitKey (0)

#-------------------------------------------------------------------------------

# ==== MAIN PROGRAM ====

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

print ("The filename to work on is %s." % sys.argv[1])

# Load image from file
im = cv2.imread (sys.argv[1])

# --- GET CONTOURS ---

# Transform image to greyscale, blur and threshold
grey = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur (grey, (5, 5), 0)

t, binary = cv2.threshold (blur, 0, 255, cv2.THRESH_BINARY
                                    + cv2.THRESH_OTSU)

# Dilate binary image to show more defined edges
kernel = numpy.ones ((9, 9), numpy.uint8)
bim = cv2.dilate(binary, kernel, iterations = 2)

# Find and draw contours
contours, hierarchy = cv2.findContours (bim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cont_im = im.copy()
cv2.drawContours (cont_im, contours, -1, (0, 0, 255), 5)

# Rectangle contours
rect_im = im.copy()

rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = numpy.int0(box)
cv2.drawContours(rect_im, [box], 0, (0, 0, 255), 2)

# Rotate and crop sub image from rectangle contour
# The following code was adapted from
# https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python 

w = rect[1][0]
h = rect[1][1]

xs = [i[0] for i in box]
ys = [i[1] for i in box]
x1 = min(xs)
x2 = max(xs)
y1 = min(ys)
y2 = max(ys)

rotated = False
angle = rect[2]

if angle < -45:
  angle += 90
  rotated = True

center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
size = (int(x2 - x1), int(y2 - y1))

M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

im_to_crop = im.copy()
cropped = cv2.getRectSubPix(im_to_crop, size, center)
cropped = cv2.warpAffine(cropped, M, size)

cropped_w = w if not rotated else h
cropped_h = h if not rotated else w

crop_im = cv2.getRectSubPix(cropped, (int(cropped_w), int(cropped_h)), (size[0] / 2, size[1] / 2))

# --- DETECT COLORS ---

# Detect green

im_hsv = cv2.cvtColor(crop_im, cv2.COLOR_BGR2HSV)
low_bd = numpy.array([50, 25, 25])
up_bd = numpy.array([70, 255, 255])
green_mask = cv2.inRange(im_hsv, low_bd, up_bd)

# Find contours and bounding box of arrow

bim_mask = cv2.erode(green_mask, kernel, iterations = 1)
hsv_contours, hsv_hierarchy = cv2.findContours(bim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

green_im = crop_im.copy()
x, y, w, h = cv2.boundingRect(hsv_contours[0])
cv2.rectangle(green_im, (x,y), (x+w,y+h), (0, 255, 0), 2)

# Get dimensions and calculate position of arrow

dims = crop_im.shape

if x < dims[1]/2:
  crop_im = cv2.rotate(crop_im, cv2.ROTATE_180)

# Detect red

im_hsv = cv2.cvtColor(crop_im, cv2.COLOR_BGR2HSV)
low_bd = numpy.array([160, 100, 20])
up_bd = numpy.array([180, 255, 255])
red_mask = cv2.inRange(im_hsv, low_bd, up_bd)

# Find and draw contours of triangle

bim_mask = cv2.dilate(red_mask, kernel, iterations = 1)

hsv_contours, hsv_hierarchy = cv2.findContours(bim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

red_im = crop_im.copy()
cv2.drawContours (red_im, hsv_contours, -1, (0, 0, 255), 5)

# --- DETERMINE ORIENTATION ---

im_line = crop_im.copy()

# Get points of triangle

angle, tri = cv2.minEnclosingTriangle(hsv_contours[0])
p1 = tri[0]
p2 = tri[1]
p3 = tri[2]

# Calculate centroid

M = cv2.moments(hsv_contours[0])
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# Get distances between centroid and points

dist1 = math.hypot(p1[0][0] - cx, p1[0][1] - cy)
dist2 = math.hypot(p2[0][0] - cx, p2[0][1] - cy)
dist3 = math.hypot(p3[0][0] - cx, p3[0][1] - cy)

# Maximum distance is direction in which triangle is pointing

max_dist = max(dist1, dist2, dist3)

if max_dist == dist1:
  point = p1
elif max_dist == dist2:
  point = p2
elif max_dist == dist3:
  point = p3

# Draw orientation line

cv2.line(im_line, (int(point[0][0]), int(point[0][1])), (cx, cy), (0, 0, 255), 2)

y = dims[0]
x = dims[1]

print(dims, point)

# Normalize and transform points

x_norm = (x - point[0][0]) / x
y_norm = (y - point[0][1]) / y

# Calculate angle and convert to degrees
# Add 90 due to point of reference being horizontal line

angle = math.atan2(int(point[0][1]) - cy, int(point[0][0] - cx))
angle = numpy.rad2deg(angle) + 90

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (1 - x_norm, y_norm))
print ("BEARING %.1f" % angle)

#-------------------------------------------------------------------------------