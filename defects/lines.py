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
    kernel = cv2.getGaborKernel((line_amplitude, mask_width), sigma=mask_width*variance/2, theta=np.pi/2, lambd=lambd, gamma=gamma)
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
              mask_width: float, frequency: int, gamma:float, variance:float, noise:bool):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = int(height*(line_amplitude-1)//100)
    mask_width = int(width*(mask_width-1)//100)
    x = location[0]*width//100
    y = location[1]*height//100
    mask = gaussian_intensity_lines(line_amplitude=line_amplitude, mask_width=mask_width, frequency=frequency, gamma=gamma, variance=variance)
    if noise:
        mask = noise4lines(mask, prob = 0.4)
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

'''
img = cv2.imread("images/image4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x, y = 22, 14
vertical = False
dark = False
brightness = 98
line_amplitude =13
mask_width = 8
frequency = 0.08
gamma = 0.2
variance = 0.59
noise = True
image = add_lines(gray, (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
#image = add_noise(gray, prob=0.9)


image = cv2.resize(image, (960, 960)) 
height, width = image.shape

lambd = frequency*mask_width*width/100
w = mask_width*width/200
n = int(w/lambd)
X_centers = []
print(n)
if n<=3 and mask_width>30:
    for i in range(n+1):
        xcp = lambd*i+lambd/4
        print(np.exp(-(xcp**2)*0.5/(variance*w)**2))
        if np.exp(-(xcp**2)*0.5/(variance*w)**2) > 0.4 and xcp<w:
            X_centers.append(xcp+w-lambd/4)

        xcm = -lambd*(i+1)+lambd/4
        print(np.exp(-(xcm**2)*0.5/((variance*w)**2)))
        if np.exp(-(xcm**2)*0.5/(variance*w)**2) > 0.4 and -xcm<w:
            X_centers.append(xcm+w-lambd/4)

    for xc in X_centers:
        cv2.rectangle(image, (int(x*width/100 + xc), int(y*height/100)), (int((x)*width/100 + xc + np.where(lambd/2>w*2-xc, w*2-xc, lambd/2)), int((y+line_amplitude)*height/100)), color=(255,255,255), thickness=2)
else:
    cv2.rectangle(image, (int(x*width/100), int(y*height/100)), (int((x)*width/100+w*2), int((y+line_amplitude)*height/100)), color=(255,255,255), thickness=2)



cv2.imshow("image", image)
#cv2.imshow("mask", mask)
cv2.waitKey(0) 
cv2.destroyAllWindows()
'''