import cv2
import numpy as np
import imutils

def add_blur(img, Xmin, Ymin, Xmax, Ymax):
    height, width = img.shape
    w = (Xmax-Xmin)*width//100
    h = (Ymax-Ymin)*height//100
    x, y = Xmin*width//100, Ymin*height//100
    x_w, y_h = Xmax*width//100, Ymax*height//100
    ksize = int(np.ceil(min(w, h)/70))
    ksize += (ksize+1)%2
    sub_img = img[y:y+y_h, x:x+x_w]
    img[y:y+y_h, x:x+x_w]= cv2.GaussianBlur(sub_img, (ksize, ksize), int(ksize/2))
    yc = y+h/2
    xc = x+w/2
    bbox = [[xc/width, yc/height, w/width, h/height]]
    return img, bbox

def planet_shift(image):
    height, width = image.shape
    gray_shade = int(np.random.uniform(30, 51))
    gray_mask = np.ones(image.shape, dtype=np.uint8)*gray_shade
    blur_mask = np.zeros(image.shape, dtype=np.uint8)
    center_coordinates = (width//2, height//2)
    radius = int(width/2.14)
    color = (0, 0, 0)
    thickness = -1
    gray_mask = cv2.circle(gray_mask, center_coordinates, radius, color, thickness)
    image +=gray_mask
    color = (1, 1, 1)
    thickness = 30
    blur_mask = cv2.circle(blur_mask, center_coordinates, radius, color, thickness)
    sub_img= image*blur_mask
    sub_img[sub_img==0] = gray_shade
    sub_img = cv2.blur(sub_img, (20, 20))
    image[blur_mask==1]=sub_img[blur_mask==1]
    vertical = np.random.rand()>=0.5
    shift = np.random.uniform(0.09, 0.11)
    if np.random.rand()>=0.5:
        sign = 1
    else:
        sign = -1
    xc = width/2
    yc = height/2 - sign*radius/2
    mw = 2*radius
    mh = radius+shift*height/((shift+2))
    px, py = 2, 2+shift*sign
    if not vertical:
        xc, yc = yc, xc
        px, py = py, px
        mw, mh = mh, mw
    center_coordinates = (int(width/px), int(height/py))
    mask = np.zeros(image.shape, dtype=np.uint8)
    color = (1, 1, 1)
    thickness = -1
    mask = cv2.circle(mask, center_coordinates, radius, color, thickness)
    image*=mask
    bbox = [[xc/width, yc/height, mw/width, mh/height]]
    return image, bbox

def double_contours(image):
    height, width = image.shape
    edged = cv2.Canny(image, 100, 450) 
    dilation = cv2.dilate(edged,(1, 1),iterations = 10)
    sub_img= image*dilation
    sub_img = cv2.blur(sub_img, (10, 10))
    image[dilation==255]=sub_img[dilation==255]//4
    cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bbox = []
    for c in cnts:
        c = np.reshape(c, (len(c), 2))
        w = np.max(np.reshape(c[:, 0], -1))-np.min(np.reshape(c[:, 0], -1))
        h = np.max(c[:, 1])-np.min(c[:, 1])
        xc = np.min(c[:, 0]) + w/2
        yc = np.min(c[:, 1]) + h/2
        bbox.append([xc/width, yc/height, w/width, h/height])
    return image, bbox

def rm_half(img):
    height, width = img.shape
    loc = np.random.randint(width/2, width*2/3)
    for y in range(height):
        xr = int(np.random.normal(0, 30))
        img[y, 0:loc+xr] = 0
    bbox = [[(loc+30)/(2*width), 0.5, loc/width, 1]]
    return img, bbox

def darken_half(img):
    height, width = img.shape
    loc = np.random.randint(width/3, width/2)
    for y in range(height):
        xr = int(np.random.normal(0, 30))
        img[y, 0:loc+xr]//=2
    bbox = [[(loc+30)/(2*width), 0.5, loc/width, 1]]
    return img, bbox

def disk_defect(image):
    arti = np.random.randint(0, 2)
    if arti ==0:
        return darken_half(image)
    else:
        return rm_half(image)

def test():
    img = cv2.imread("images/image2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image, _ = double_contours(gray)

    image = cv2.resize(image, (960, 960)) 
    cv2.imshow("image", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

#test()
