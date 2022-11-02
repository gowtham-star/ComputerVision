import numpy as np
import cv2

imgPath=["Test1/testL.jpg","Test1/testC.jpg","Test1/testR.jpg"]

imgPath2=["Test2/bookstore1.jpg","Test2/bookstore2.jpg","Test2/bookstore3.jpg"]

imgPath3=["Test3/bookstore5.jpg","Test3/bookstore6.jpg","Test3/bookstore7.jpg"]

imgPath4=["Test4/sc1.jpg","Test4/sc2.jpg","Test4/sc3.jpg"]

imgPath5=["Test5/ul1.jpg","Test5/ul2.jpg","Test5/ul3.jpg"]

imagePathArr = [imgPath,imgPath2,imgPath3,imgPath4,imgPath5]
images = []

j=1
for i in imagePathArr:	
	images = []
	for path in i:
		image = cv2.imread(path)
		images.append(image)
	print(len(images))
	stitcher = cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch(images)

	if status == 0:
		print("Image Stitching Successful")
		cv2.imshow("Stitched Image", stitched)
		cv2.imwrite("stichedSet" + str(j) +  ".png",stitched)
	else:
		print("[INFO] Failed Image Stitching ({})".format(status))
	j+=1

