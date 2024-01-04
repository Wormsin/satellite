import cv2
import numpy as np

def noise_lines(line_amplitude:int, mask_width: int, num: int, blur:bool):
    mask = np.zeros((mask_width, line_amplitude))

    line_space = mask_width//num
    line_width = line_space//2
    for i in range(0, mask.shape[0], line_space):
        mask[i:i+line_width]+=1
    
    if blur:
        mask = cv2.GaussianBlur(mask,ksize=(line_width//2,line_width+1),sigmaX=line_width//4,sigmaY=line_width//4)
    return mask

def gaussian_intensity_lines(line_amplitude:int, mask_width: int, frequency: int, gamma:float, variance:float):
    lambd = mask_width*frequency
    kernel = cv2.getGaborKernel((line_amplitude, mask_width), sigma=mask_width*variance, theta=np.pi/2, lambd=lambd, gamma=gamma)
    return kernel

def noise4lines(mask, prob = 0.5):
    noise = np.random.normal(0, 0.1, size=mask.shape)
    noise[noise>prob-0.5] = 0
    noise[noise<prob-0.5] = 1
    mask[noise==1] = 0
    return mask

def add_noise(img, prob=0.5):
    noise = np.random.normal(0, 0.1, size=img.shape)
    noise[noise>prob-0.5] = 1
    noise[noise<prob-0.5] = 0
    img[noise==1] = 255
    return img

def add_lines(img: np.uint, location:tuple, vertical: bool, dark:bool, brightness: int, line_amplitude:float, 
              mask_width: float, frequency: int, gamma:float, variance:float):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = int(height*(line_amplitude-1)//100)
    mask_width = int(width*(mask_width-1)//100)
    x = location[0]*width//100
    y = location[1]*height//100
    mask = gaussian_intensity_lines(line_amplitude=line_amplitude, mask_width=mask_width, frequency=frequency, gamma=gamma, variance=variance)
    mask = noise4lines(mask, prob = 0.6)
    mask[mask<0.001] = 0
    if vertical:
        mask = mask.T
    line_y, line_x = mask.shape
    
    if not dark:
        bw_mask = ((mask*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask
        bw_mask[bw_mask<0] =0 
        bw_mask = bw_mask.astype(np.uint8)
        img[y:y+line_y, x:x+line_x]+=bw_mask
    else:
        bw_mask = (((1-mask)*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask*(-1)
        bw_mask[bw_mask<0] =0 
        bw_mask = bw_mask.astype(np.uint8)
        img[y:y+line_y, x:x+line_x]-=bw_mask
    
    img[image0==0] = 0
    return img


img = cv2.imread("images/image4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#image = add_lines(img=gray, location=(0, 0), vertical=False, dark=False, brightness=100, line_amplitude=100, mask_width=100, frequency=1, gamma=0, variance=1)
#mask = gaussian_intensity_lines(line_amplitude=50, mask_width=20, frequency=0.18, gamma=0.4, variance=0.4)
image = add_noise(gray, prob=0.9)

image = cv2.resize(image, (960, 960)) 
cv2.imshow("image", image)
#cv2.imshow("mask", mask)
cv2.waitKey(0) 
cv2.destroyAllWindows()