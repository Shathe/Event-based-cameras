import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt








def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


for imagePath in glob.glob('./Event*/*/*/*'):
		# load the image, convert it to grayscale, and blur it slightly
	print(imagePath)
	try:
	    os.remove(imagePath)
	except OSError:
	    pass


for imagePath in glob.glob('./Caltech101-Normal2/TRAIN/*/*'):
		# load the image, convert it to grayscale, and blur it slightly
	if '_5' in imagePath:
		print(imagePath)
		image = cv2.imread(imagePath)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray[gray>20]=255

		kernel2 = np.ones((2, 2),np.uint8)
		dilation = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
		blur2 = cv2.blur(dilation,(2,2))
		blur2[blur2<100]=0
		blur2[blur2>=100]=255

		path = imagePath.replace('Caltech101-Normal2','Event2_Caltech101')

		# show the images
		print(path)
		cv2.imwrite(path,blur2)


for imagePath in glob.glob('./RGB_Caltech101/TRAIN/*/*'):
		# load the image, convert it to grayscale, and blur it slightly
	print(imagePath)
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	wide = cv2.Canny(blurred, 90, 210)
	auto = auto_canny(blurred, 0.30)
	kernel = np.ones((3, 3),np.uint8)
	kernel2 = np.ones((2 , 2),np.uint8)
	dilation = cv2.dilate(auto,kernel,iterations = 1)
	dilation3 = cv2.dilate(auto,kernel2,iterations = 1)

	path = imagePath.replace('RGB_','Canny_')

	path1 = path.replace(' ',' ')
	path2 = path.replace('.jpg','can1.jpg')
	path3 = path.replace('.jpg','can2.jpg')
	path4 = path.replace('.jpg','can3.jpg')
	print(path1)

	wide[wide<100]=0
	wide[wide>=100]=255
	dilation[dilation<100]=0
	auto[auto>=100]=255
	auto[auto<100]=0
	dilation[dilation>=100]=255
	dilation3[dilation3<100]=0
	dilation3[dilation3>=100]=255


	cv2.imwrite(path1,wide)
	cv2.imwrite(path2,auto)
	cv2.imwrite(path3,dilation3)
	cv2.imwrite(path4,dilation)

for imagePath in glob.glob('./Caltech101-Big2/TRAIN/*/*'):
		# load the image, convert it to grayscale, and blur it slightly
	print(imagePath)
	image = cv2.imread(imagePath)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = np.ones((3, 3),np.uint8)
	kernel2 = np.ones((2, 2),np.uint8)
	gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel2)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)

	gray1 = gray.copy()
	gray1[gray<50]=0
	gray1[gray>=50]=255

	path = imagePath.replace('Caltech101-Big2','Event_Caltech101')

	path1 = path.replace(' ',' ')
	# show the images
	print(path1)
	cv2.imwrite(path1,gray1)


