import cv2
import numpy as np

def add_blur(img, Xmin, Ymin, Xmax, Ymax):
    height, width = img.shape
    w = (Xmax-Xmin)*width/100
    h = (Ymax-Ymin)*height/100
    x, y = Xmin*width/100, Ymin*height/100
    x_w, y_h = Xmax*width/100, Ymax*height/100
    ksize = int(np.ceil(min(w, h)/70))
    ksize += (ksize+1)%2
    sub_img = img[y:y+y_h, x:x+x_w]
    img[y:y+y_h, x:x+x_w]= cv2.GaussianBlur(sub_img, (ksize, ksize), int(ksize/2))
    return img

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
    px, py = 2, 2+shift*sign
    if not vertical:
        px, py = py, px
    center_coordinates = (int(width/px), int(height/py))
    mask = np.zeros(image.shape, dtype=np.uint8)
    color = (1, 1, 1)
    thickness = -1
    mask = cv2.circle(mask, center_coordinates, radius, color, thickness)
    image*=mask
    return image

def double_contours(image):
    edged = cv2.Canny(image, 100, 450) 
    dilation = cv2.dilate(edged,(1, 1),iterations = 10)
    sub_img= image*dilation
    sub_img = cv2.blur(sub_img, (10, 10))
    image[dilation==255]=sub_img[dilation==255]//4
    return image

def rm_half(img):
    height, width = img.shape
    loc = np.random.randint(width/2, width*2/3)
    for y in range(height):
        xr = int(np.random.normal(0, 30))
        img[y, 0:loc+xr] = 0
    return img

def darken_half(img):
    height, width = img.shape
    loc = np.random.randint(width/3, width/2)
    for y in range(height):
        xr = int(np.random.normal(0, 30))
        img[y, 0:loc+xr]//=2
    return img

def disk_defect(image):
    arti = np.random.randint(0, 2)
    if arti ==0:
        return darken_half(image)
    else:
        return rm_half(image)

def test():
    img = cv2.imread("images/image8.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = disk_defect(gray)

    image = cv2.resize(image, (960, 960)) 
    cv2.imshow("image", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

#test()