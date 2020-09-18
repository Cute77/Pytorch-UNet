from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

noise_fraction = 0.2
inputdir = 'ISIC-2017_Training_Part1_GroundTruth'
outputdir = inputdir + '_' + str(noise_fraction) + '/'

for filename in listdir(inputdir):
    img = cv2.imread(inputdir+'/'+filename)
    print(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = noise_fraction * 0.1 * cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
        # cv2.drawContours(black, [approx1], -1, (255, 255, 0), 2)
        # cv2.drawContours(black, [approx2], -1, (0, 0, 255), 2)

    # cv2.imwrite('00_original.jpg', img)

    black = cv2.cvtColor(np.zeros((int(img.shape[0]), int(img.shape[1])), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.fillPoly(black, [approx], (255,255,255))
    path = outputdir + filename
    cv2.imwrite(path, black)



'''
img = cv2.imread('ISIC_0000000_segmentation.png')
# img = cv2.resize(img, (512, 512))

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    epsilon1 = 0.02 * cv2.arcLength(cnt, True)
    epsilon2 = 0.04 * cv2.arcLength(cnt, True)

    approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
    approx2 = cv2.approxPolyDP(cnt, epsilon2, True)

    # cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # cv2.drawContours(black, [approx1], -1, (255, 255, 0), 2)
    # cv2.drawContours(black, [approx2], -1, (0, 0, 255), 2)

# cv2.imwrite('00_original.jpg', img)

black = cv2.cvtColor(np.zeros((int(img.shape[0]), int(img.shape[1])), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
cv2.fillPoly(black, [approx1],(255,255,255))
cv2.imwrite('00_0.2.jpg', black)
ax2.imshow(black)
ax2.set_title('noise-fraction_0.2')

black = cv2.cvtColor(np.zeros((int(img.shape[0]), int(img.shape[1])), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
cv2.fillPoly(black, [approx2], (255,255,255))
cv2.imwrite('00_0.4.jpg', black)
ax3.imshow(black)
ax3.set_title('noise-fractio_0.4')


x1 = img.copy()
epsilon1 = 50 
approx = cv2.approxPolyDP(contours[0],epsilon,True)
cv2.polylines(x1, [approx], True, (0, 0, 255), 2)
# cv2.putText(x1, "epsilon:50" , (160,180), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2 )
cv2.imwrite('approxcurve1.jpg' , x1 ) 
x1 = img.copy()

epsilon = 30 
approx = cv2.approxPolyDP(contours[0],epsilon,True)
cv2.polylines(x1, [approx], True, (255, 0, 0), 2)
# cv2.putText(x1, "epsilon:30" , (160,180), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2 )
cv2.imwrite('approxcurve2.jpg' , x1 ) 

x1 = img.copy()
epsilon = 35 
approx = cv2.approxPolyDP(contours[0],epsilon,True)
cv2.polylines(x1, [approx], True, (0, 255, 0), 2)
# cv2.putText(x1, "epsilon:35" , (160,180), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2 )
cv2.imwrite( 'approxcurve3.jpg' , x1) 

fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1, ax3= axes.ravel()
ax0.imshow(img)
ax0.set_title('original image')

ax1.imshow(chull)
ax1.set_title('convex_hull image')
'''