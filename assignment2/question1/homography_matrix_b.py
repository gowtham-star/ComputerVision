import cv2
import numpy as np
import matplotlib.pyplot as plt
#This script generates homography matrix
src_points = np.array([[2038,1439],[2920,1440],[2025,1982],[2879,2002],[2425,1996]])
dest_points = np.array([[268,1305],[1136,1312],[317,1872],[1156,1831],[772,1856]])
h, status = cv2.findHomography(src_points, dest_points)
print("Homography Matrix:" + str(h))
im_src = cv2.imread('images/img1.png')
im_dst = cv2.imread('images/img2.png')
#this will give warped image output using h as homography matrix
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
cv2.imshow("Warped_Source_Image", im_out)
plt.imshow(im_out)
plt.show()