from shutil import copy, rmtree
import numpy as np
import albumentations as A



def randv_freq_lines():
    noise = False
    vertical, dark = np.random.rand(2)>=0.5
    x, y = np.random.uniform(0.3, 0.65, 2)
    line_amplitude = int(np.random.uniform(0.05, 1-y)*100)
    mask_width = int(np.random.uniform(0.1, 0.3)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.7, 1)*100)
    frequency = np.random.uniform(0.01, 0.05)
    variance = np.random.uniform(0.6, 1)
    gamma = np.random.uniform(0.3, 0.6)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def randv_middle_lines():
    noise = False
    vertical, dark = np.random.rand(2)>=0.5
    x, y = np.random.uniform(0.1, 0.65, 2)
    line_amplitude = int(np.random.uniform(0.05, 1-y)*100)
    mask_width = int(np.random.uniform(0.3, 1-x)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.7, 1)*100)
    frequency = np.random.uniform(0.01, 0.1)
    variance = np.random.uniform(0.6, 1)
    gamma = np.random.uniform(0.3, 0.6)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def randv_single_middle_lines():
    noise = False
    vertical, dark = np.random.rand(2)>=0.5
    x, y = np.random.uniform(0.1, 0.65, 2)
    line_amplitude = int(np.random.uniform(0.05, 1-y)*100)
    mask_width = int(np.random.uniform(0.05, 0.25)*100)
    if not vertical:
        x, y = y, x
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.7, 1)*100)
    frequency = np.random.uniform(0.08, 0.5)
    variance = np.random.uniform(0.8, 1)
    gamma = np.random.uniform(0.3, 0.6)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

def randv_mixed_lines(conditions):
    Xmin, Ymin, Xmax, Ymax = conditions
    noise = False
    vertical, dark = np.random.rand(2)>=0.5
    if not vertical:
        Xmin, Ymin, Xmax, Ymax = Ymin, Xmin, Ymax, Xmax
    x = np.random.uniform(Xmin/100, Xmax/100-0.2)
    y  = np.random.uniform(Ymin/100, Ymax/100-0.2)
    line_amplitude = int(np.random.uniform(0.05, Ymax/100-y)*100)
    mask_width = int(np.random.uniform(0.2, Xmax/100-x)*100)
    y = int(y*100)
    x = int(x*100)
    brightness = int(np.random.uniform(0.8, 1)*100)
    frequency = np.random.uniform(0.5, 0.8)
    variance = 1
    gamma = np.random.uniform(0, 0.02)
    return (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise

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
    brightness = int(np.random.uniform(0.4, 1)*100)
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
    brightness = int(np.random.uniform(0.4, 1)*100)
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
