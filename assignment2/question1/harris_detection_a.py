import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("images/img1.png")
grayImg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
grayImg = np.float32(grayImg)
dest = cv2.cornerHarris(grayImg,8, 29, 0.05)
dest = cv2.dilate(dest, None)
# Reverting back to the original image with optimal threshold value
image[dest > 0.01 * dest.max()]=[0, 0, 255]
cv2.imshow('Image with corners', image)
cv2.imwrite("img1C.png",image)
plt.imshow(image)
plt.show()