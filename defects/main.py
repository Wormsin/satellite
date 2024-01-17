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

def make_bbox(x, y, mask_width, line_amplitude, vertical, variance, frequency, gamma):
    lambd = frequency*mask_width
    w = mask_width/2
    n = int(w/lambd)+1
    Xc = x+mask_width*vertical/2 +(not vertical)*line_amplitude/2
    Yc = y + line_amplitude*vertical/2 + (not vertical)*mask_width/2
    mask_width = np.where(variance<0.8, variance*mask_width*1.1, mask_width)
    line_amplitude = np.where(gamma-variance>0.2, variance*line_amplitude/gamma, line_amplitude)
    width = lambd*vertical/2 + line_amplitude*(not vertical)
    height = line_amplitude*vertical + lambd*(not vertical)/2
    bbox = []
    if n<=5 and mask_width>10:
        for i in range(n+1):
            cp = lambd*i+lambd/4
            if np.exp(-(cp**2)*0.5/(variance*w*2)**2) >= 0.45 and cp<mask_width/2:
                #X_centers.append(xcp)
                xb = (Xc+cp)*vertical+Xc*(not vertical)
                yb = Yc*vertical+(Yc+cp)*(not vertical)
                bbox.append([xb/100, yb/100, width/100, height/100])
            cm = -lambd*i+lambd/4
            if np.exp(-(cm**2)*0.5/(variance*w*2)**2) >= 0.45 and -cm<mask_width/2:
                xb = (Xc+cm)*vertical+Xc*(not vertical)
                yb = Yc*vertical+(Yc+cm)*(not vertical)
                bbox.append([xb/100, yb/100, width/100, height/100])
    else:
        if not vertical:
            mask_width, line_amplitude = line_amplitude, mask_width
        bbox.append([Xc/100, Yc/100, mask_width/100, line_amplitude/100])
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
    if num <=5:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_big_lines()
    elif num >5 and num <=8:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_packages()
    else:
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = randv_destroyed_pixels()

    bbox_arr = make_bbox(location[0], location[1], mask_width, line_amplitude, vertical, variance, frequency, gamma)
    defected_img = add_lines(gray, location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    save_data(num, defected_img, bbox_arr, 0)

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
