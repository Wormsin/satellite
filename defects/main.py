import os
from shutil import copy, rmtree
import cv2
import numpy as np
from lines import add_lines
import albumentations as A

def save_data(num, image, bbox_arr, cls):
    name = str(format((num+1)/1000000, '6f'))
    img_path = os.path.join(IMG_PATH, f'{name[2:]}.jpg')
    bbox_path = os.path.join(LABELS_PATH, f'{name[2:]}.txt')
    cv2.imwrite(img_path, image)
    if len(bbox_arr)!=0:
        for bbox in bbox_arr:
            line = (cls, *bbox)
            with open(bbox_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
    else:
        file = open(bbox_path, 'a')
        file.close()

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
    gamma = np.random.uniform(0, 0.6)
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

def augment(image, bboxes, cls):
    class_labels = np.zeros(len(bboxes))
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=1),
        A.RandomCrop(width=560, height=560, p=0.4),
    ],
    bbox_params=A.BboxParams(format='yolo', min_area=25, label_fields=['class_labels']),
    )
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    return transformed_image, transformed_bboxes

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
                xb = (Xc+cp)*vertical+Xc*(not vertical)
                yb = Yc*vertical+(Yc+cp)*(not vertical)
                bbox.append([xb/100, yb/100, width/100, height/100])
            cm = -lambd*(i+1)+lambd/4
            if np.exp(-(cm**2)*0.5/(variance*w*2)**2) >= 0.45 and -cm<mask_width/2:
                xb = (Xc+cm)*vertical+Xc*(not vertical)
                yb = Yc*vertical+(Yc+cm)*(not vertical)
                bbox.append([xb/100, yb/100, width/100, height/100])
    else:
        if not vertical:
            mask_width, line_amplitude = line_amplitude, mask_width
        bbox.append([Xc/100, Yc/100, mask_width/100, line_amplitude/100])
    return bbox

def process_img(image, defect_func, num, num_cycles_defects, augmentation, display):
    bbox_arr = []
    for j in range(num_cycles_defects):
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = defect_func()
        bbox_arr+= make_bbox(location[0], location[1], mask_width, line_amplitude, vertical, variance, frequency, gamma)
        image = add_lines(image, location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    if display: display_img(image, bbox_arr)
    save_data(num, image, bbox_arr, 0)
    for i in range(augmentation):
        aug_img, aug_bboxes = augment(image, bboxes=bbox_arr, cls=0)
        if display: display_img(aug_img, aug_bboxes)
        save_data(num+1+i, aug_img, aug_bboxes, 0)
    return augmentation

def display_img(image, bbox_arr):
    img = cv2.resize(image, (960, 960)) 
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

IMG_PATH = 'dataset/images/train'
LABELS_PATH = 'dataset/labels/train'
SOURCE = "validation_data"

def main():
    name_shift = 0
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        os.makedirs(LABELS_PATH)
    else:
        filenames = os.listdir('dataset/labels/train')
        if len(filenames)>0:
            filenames = np.array([file.rstrip(".txt") for file in filenames])
            name_shift = np.max(np.int32(filenames))

    num_images = len(os.listdir(SOURCE))//3
    for num, filename in enumerate(os.listdir(SOURCE)):
        num+=name_shift
        print(num-name_shift)
        image = cv2.imread(os.path.join(SOURCE, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 3000:
            gray = cv2.resize(gray, (1080, 1080)) 

        #name_shift += process_img(gray, randv_big_lines, num, 0, 2, True)
        
        if num<=num_images:
            num_cycles_defects = np.random.randint(1, 3)
            name_shift += process_img(gray, randv_big_lines, num, num_cycles_defects, 0, False)
        elif num>num_images and num<=num_images*2:
            num_cycles_defects = np.random.randint(1, 5)
            name_shift += process_img(gray, randv_packages, num, num_cycles_defects, 0, False)
        else:
            num_cycles_defects = np.random.randint(1, 5)
            name_shift += process_img(gray, randv_destroyed_pixels, num, num_cycles_defects, 0, False)


if __name__ == "__main__":
    main()