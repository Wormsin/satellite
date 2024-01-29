import cv2
import numpy as np
import os
from lines import multiple_stripes, blur_lines, dark_noise_lines, striped_lines, basic_lines
from params import make_bbox, augment
from big_defects import add_blur, planet_shift, double_contours, disk_defect


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

def process_lines_art(image, defect_func, type_lines_func, blur, num, num_cycles_defects, augmentation, display, conditions = []):
    bbox_arr = []
    for j in range(num_cycles_defects):
        location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise = defect_func(conditions) if conditions!=[] else defect_func()
        bbox_arr+= make_bbox(location[0], location[1], mask_width, line_amplitude, vertical, variance, frequency, gamma)
        image = type_lines_func(image, location, vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    if blur: 
        height, width = image.shape
        Xmin, Ymin, Xmax, Ymax =  conditions[0]*width/100, conditions[1]*height/100, conditions[2]*width/100, conditions[3]*height/100
        image, bbox_arr = add_blur(image, Xmin, Ymin, Xmax, Ymax)
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

def process_big_art(image, defect_func, num, augmentation, display):
    image, bbox_arr = defect_func(image)
    if display: display_img(image, bbox_arr)
    save_data(num, image, bbox_arr, 0)
    for i in range(augmentation):
        aug_img, aug_bboxes = augment(image, bboxes=bbox_arr, cls=0)
        if display: display_img(aug_img, aug_bboxes)
        save_data(num+1+i, aug_img, aug_bboxes, 0)
    return augmentation


IMG_PATH = 'dataset/images/train'
LABELS_PATH = 'dataset/labels/train'
SOURCE = "images"

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

        name_shift += process_big_art(gray, double_contours, num, 0, True)
        
        '''
        if num<=num_images:
            num_cycles_defects = np.random.randint(1, 3)
            name_shift += process_img(gray, randv_big_lines, num, num_cycles_defects, 0, False)
        elif num>num_images and num<=num_images*2:
            num_cycles_defects = np.random.randint(1, 5)
            name_shift += process_img(gray, randv_packages, num, num_cycles_defects, 0, False)
        else:
            num_cycles_defects = np.random.randint(1, 5)
            name_shift += process_img(gray, randv_destroyed_pixels, num, num_cycles_defects, 0, False)
        '''

if __name__ == "__main__":
    main()