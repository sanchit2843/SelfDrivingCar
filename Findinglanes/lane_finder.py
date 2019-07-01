import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.imread('test.jpg',0)
plt.imshow(image)
kernel_size = 5
blur_gray = cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
edges = cv2.Canny(blur_gray, 50,150)

# Display the image
plt.imshow(edges, cmap='Greys_r')


mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)