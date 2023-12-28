import cv2
import numpy as np
import time
import os

def print_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)                         

img = cv2.imread('image.jpg')

hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lower_range = np.array([0,0,200], dtype=np.uint8)
upper_range = np.array([255,40,255], dtype=np.uint8)
bw_img = cv2.inRange(hsv_image, lower_range, upper_range)

mask = np.zeros((img.shape[0], img.shape[1]))
lines = cv2.HoughLinesP(bw_img, rho=1, theta=np.pi/2, minLineLength=170, threshold=200)
for i, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    cv2.line(mask, (x1, y1), (x2, y2), (255), 1)

print_img("bw_img", bw_img)
print_img("mask", mask)
print_img("result", img)
cv2.destroyAllWindows()
print(lines.shape)


if not os.path.exists("masks"):
        os.mkdir("masks")
with open("masks/mask.txt", 'a') as f:
    np.savetxt(f, mask)
