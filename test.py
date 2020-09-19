from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'ISIC_0000050_segmentation.png'
noise_fraction = 0.03

img = cv2.imread(path)
print(img.shape)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    epsilon = noise_fraction * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

# black = cv2.cvtColor(np.zeros((int(img.shape[0]), int(img.shape[1])), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
# cv2.fillPoly(black, [approx], (255, 255, 255))
black = np.zeros((int(img.shape[0]), int(img.shape[1])), dtype=np.uint8)
cv2.fillPoly(black, [approx], 255)
# black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
print(black.shape)
output = str(noise_fraction) + '_' + path
cv2.imwrite(output, black)