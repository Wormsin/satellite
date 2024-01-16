import os
from shutil import copy, rmtree
import cv2
import numpy as np
from lines import add_lines

def save_data(num, image, bbox_arr, cls):
    name = str(format((num+1)/1000000, '6f'))
    img_path = os.path.join(IMG_PATH, f'{name[2:]}.jpg')
    bbox_path = os.path.join(LABELS_PATH, f'{name[2:]}.txt')
    cv2.imwrite(img_path, image)
    for bbox in bbox_arr:
        line = (cls, *bbox)
        with open(bbox_path, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def randv_big_lines():
    noise = False
    vertical, dark = np.random.rand(2)>=0.5
    x = np.random.uniform(0, 0.7)
    y  = 0.02
    line_amplitude = 95
    mask_width = int(np.random.uniform(0.05, 1-x)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.5, 1)*100)
    frequency = np.random.uniform(0.01, 0.8)
    variance = np.random.uniform(0.3, 1)
    gamma = np.random.uniform(0, 0.02)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def randv_packages():
    vertical, dark, noise = np.random.rand(3)>=0.5
    x, y = np.random.uniform(0.1, 0.8, 2)
    line_amplitude = int(np.random.uniform(0.05, 0.2)*100)
    mask_width = int(np.random.uniform(0.05, 1-x)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.5, 1)*100)
    frequency = np.random.uniform(0.01, 0.2)
    variance = np.random.uniform(0.3, 1)
    gamma = np.random.uniform(0, 0.4)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def randv_destroyed_pixels():
    vertical, dark, noise = np.random.rand(3)>=0.5
    x, y = np.random.uniform(0.1, 0.8, 2)
    line_amplitude = int(np.random.uniform(0.2, 1-y)*100)
    mask_width = int(np.random.uniform(0.05, 0.1)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.8, 1)*100)
    frequency = np.random.uniform(0.01, 0.1)
    variance = 1
    gamma = np.random.uniform(0, 0.2)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def make_bbox(x, y, mask_width, line_amplitude, vertical, variance):
    lambd = frequency*mask_width/100
    w = mask_width/200
    n = int(w/lambd)
    X_centers = []
    if n<=3 and mask_width>30:
        for i in range(n+1):
            xcp = lambd*i+lambd/4
            if np.exp(-(xcp**2)*0.5/(variance*w)**2) > 0.4 and xcp<w:
                X_centers.append(xcp+w)
            xcm = -lambd*(i+1)+lambd/4
            if np.exp(-(xcm**2)*0.5/(variance*w)**2) > 0.4 and -xcm<w:
                X_centers.append(xcm+w)
    
    if len(X_centers)==0:
        width = mask_width/100
        height = line_amplitude/100
        if not vertical:
            height, width = width, height
        x_center = x/100+width/2
        y_center = y/100+height/2
        return [[x_center, y_center, width, height]]
    else:
        bbox = []
        for c in X_centers:
            if not vertical:
                y_center = c+y/100
                x_center = x/100+line_amplitude/200
                width = line_amplitude/100
                height = lambd/2
            else:
                x_center = c + x/100
                y_center = y/100 + line_amplitude/200
                height = line_amplitude/100
                width = lambd/2
            bbox.append([x_center, y_center, width, height])
    return bbox


IMG_PATH = 'dataset/images/train'
LABELS_PATH = 'dataset/labels/train'
SOURCE = "images"


if os.path.exists(IMG_PATH):
        rmtree("dataset")
os.makedirs(IMG_PATH)
os.makedirs(LABELS_PATH)

for num, filename in enumerate(os.listdir(SOURCE)):
    image = cv2.imread(os.path.join(SOURCE, filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if num <=250:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_big_lines()
    elif num >250 and num <=480:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_packages()
    else:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_destroyed_pixels()

    bbox_arr = make_bbox(location[0], location[1], mask_width, line_amplitude, vertical, variance)
    defected_img = add_lines(gray, location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    save_data(num, defected_img, bbox_arr, 0)
    
    '''
    img = cv2.resize(defected_img, (960, 960)) 
    height, width = img.shape
    for bbox in bbox_arr:
        x_center = int(bbox[0]*width)
        y_center  = int(bbox[1]*height)
        m_width = int(bbox[2]*width)
        m_height = int(bbox[3]*height)
        cv2.rectangle(img, (int(x_center-m_width/2), int(y_center-m_height/2)), (int(x_center+m_width/2), int(y_center+m_height/2)), color=(255,255,255), thickness=2)
        cv2.imshow("image", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    print(location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    '''