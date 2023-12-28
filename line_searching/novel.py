import cv2
import numpy as np
import time
import os 


def print_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)                         

def calc_variance(arr):
    variance = np.var(arr, axis=0)
    means = np.mean(arr, axis=0)
    variance[means<30] = 100
    return variance

def correlation(img, l_part, precision, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_height, img_width, ch = img.shape
    gray[gray==0]=1
    print_img("gray", gray)

    l_min = int(np.floor(img_height*l_part)-np.floor(img_height*l_part)%precision)
    add_pixels = int((2*img_height)%l_min)

    add_lines = np.ones((add_pixels, img_width), np.uint8)
    gray = np.concatenate((gray, add_lines))
    print_img("extra boundaries", gray)

    mask = np.zeros((img_height, img_width))
    for li in range(0, img_height+add_pixels, l_min//precision):
        sub_img = gray[li:l_min+li]
        variance = np.sqrt(calc_variance(sub_img))
        mask[li:l_min+li, variance.T<=threshold] = 255
        img[li:l_min+li, variance.T<=threshold] = [255, 0, 255]
    return mask, img


img  = cv2.imread("image.jpg")
mask, result = correlation(img, 0.15, 20, 3)
print_img("mask", mask)
print_img("result", result)
cv2.destroyAllWindows()

if not os.path.exists("masks"):
        os.mkdir("masks")
with open("masks/mask.txt", 'a') as f:
    np.savetxt(f, mask)


