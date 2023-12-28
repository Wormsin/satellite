import cv2
import numpy as np
import time
import os 

def print_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)                         

def pix_color_change(h_axis, artifacts, mask):
    start, end = artifacts
    mask[start:end+1, h_axis] = 255
    return mask 


def rm_background(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold_img = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    mask = threshold_img
    img = cv2.bitwise_and(frame, frame, mask=mask)
    return img 

def k_means(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
    k = 15
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)
    return segmented_image


def main():
    #start = time.time()
    frame = cv2.imread('image.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    segment_img = k_means(frame)
    print_img("segmented", segment_img)
    img = rm_background(segment_img)
    print_img("without a background", img)
    img_height, img_width, ch = frame.shape

    mask = np.zeros((img_height, img_width))

    for line in range(img_width):
        pixels_line = img[:, line]
        artifact = []
        for i, p in enumerate(pixels_line):
            if i!=img_height-1:
                p_next = np.array(pixels_line[i+1], dtype=int)
                p = np.array(p, dtype=int)
                dp = np.absolute(p-p_next)
                similar = (sum(dp==0) == 3)
            else:
                similar = False
            if similar and sum(p)!=0 :
                if len(artifact)==0:
                    artifact.append(i)
                elif artifact[-1]!=i: 
                    artifact.append(i)
                artifact.append(i+1)
            else:
                if len(artifact)>=img_height*0.15:
                    mask = pix_color_change(line, (artifact[0], artifact[-1]), mask)
                artifact = []
    #end = time.time()
    #print(end-start)
    print_img("result mask", mask)
    frame[mask!=0] = np.array([255, 0, 255])
    print_img("mask on img", frame)
    cv2.destroyAllWindows()
    if not os.path.exists("masks"):
        os.mkdir("masks")
    with open("masks/mask.txt", 'a') as f:
        np.savetxt(f, mask)
               
main()




