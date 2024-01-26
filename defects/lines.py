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
              mask_width: float, frequency: int, gamma:float, variance:float, noise:bool, blur:bool):
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

def mixed_mask(img: np.uint, location:tuple, vertical: bool, dark:bool, brightness: int, line_amplitude:float, 
              mask_width: float, frequency: int, gamma:float, variance:float, noise:bool, blur:bool):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = int(height*(line_amplitude-1)//100)
    mask_width = int(width*(mask_width-1)//100)
    x = location[0]*width//100
    y = location[1]*height//100
    mask1 = gaussian_intensity_lines(line_amplitude=line_amplitude, mask_width=mask_width, frequency=frequency, gamma=gamma, variance=variance)
    mask2 = gaussian_intensity_lines(line_amplitude=mask_width, mask_width=line_amplitude, frequency=frequency/100, gamma=gamma, variance=variance)
    mask1 = mask1.T
    mask2+=1
    mask2[mask2==2] = 0
    mask = mask1*mask2
    mask[mask<0] = 0
    mask[mask>1] = 1
    cv2.imshow("mask1", mask)
    cv2.waitKey(0) 
    line_y, line_x = mask.shape

    bw_mask = ((mask*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask
    bw_mask[bw_mask<0] =0 
    bw_mask = bw_mask.astype(np.uint8)
    img[y:y+line_y, x:x+line_x]+=bw_mask
    
    mask+=1
    mask[mask>1.2]=0
    mask1[mask1<0.5] = 0
    mask1[mask1>0.5] = 1
    mask*=mask1
    mask[mask>0.5] = 1
    if blur:
        ksize = int(np.ceil(min(mask_width, line_amplitude)/100))
        mask = cv2.blur(mask, (ksize, ksize))
    cv2.imshow("mask2", mask)
    cv2.waitKey(0) 

    bw_mask = (((1-mask)*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask*(-1)
    bw_mask[bw_mask<0] =0 
    bw_mask = bw_mask.astype(np.uint8)
    img[y:y+line_y, x:x+line_x]-=bw_mask

    img[image0==0] = 0
    return img

def dark_noise(img: np.uint, location:tuple, vertical: bool, dark:bool, brightness: int, line_amplitude:float, 
              mask_width: float, frequency: int, gamma:float, variance:float, noise:bool):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = int(height*(line_amplitude-1)//100)
    mask_width = int(width*(mask_width-1)//100)
    x = location[0]*width//100
    y = location[1]*height//100
    mask= gaussian_intensity_lines(line_amplitude=line_amplitude, mask_width=mask_width, frequency=frequency, gamma=gamma, variance=variance)
    mask0 = np.copy(mask)
    if noise:
        mask = noise4lines(mask, prob = 0.4)
    mask[mask<0.001] = 0
    if vertical:
        mask = mask.T
        mask0 = mask0.T
    line_y, line_x = mask.shape

    bw_mask = ((mask*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask
    bw_mask[bw_mask<0] =0 
    bw_mask = bw_mask.astype(np.uint8)
    img[y:y+line_y, x:x+line_x]+=bw_mask

    mask2 = mask + 1
    mask2[mask2>1.2]=0
    mask2*=mask0
    mask2[mask2<0.5] = 0
    cv2.imshow("mask2", mask2)
    cv2.waitKey(0)

    bw_mask = (((1-mask2)*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask2*(-1)
    bw_mask[bw_mask<0] =0 
    bw_mask = bw_mask.astype(np.uint8)
    img[y:y+line_y, x:x+line_x]-=bw_mask

    img[image0==0] = 0
    return img

def blur_lines(img: np.uint, location:tuple, vertical: bool, dark:bool, brightness: int, line_amplitude:float, 
              mask_width: float, frequency: int, gamma:float, variance:float, noise:bool):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = int(height*(line_amplitude-1)//100)
    mask_width = int(width*(mask_width-1)//100)
    x = location[0]*width//100
    y = location[1]*height//100
    mask= gaussian_intensity_lines(line_amplitude=line_amplitude, mask_width=mask_width, frequency=frequency, gamma=gamma, variance=variance)
    if vertical:
        mask = mask.T
    line_y, line_x = mask.shape

    ksize = int(np.ceil(min(mask_width, line_amplitude)/10))
    ksize += (ksize+1)%2
    sub_img = img[y:y+line_y, x:x+line_x]
    sub_img[mask>0.5] = cv2.GaussianBlur(sub_img, (ksize, ksize), int(ksize/2))[mask>0.5]
    img[y:y+line_y, x:x+line_x] = sub_img

    cv2.imshow("mask1", sub_img)
    cv2.waitKey(0)
    img[image0==0] = 0
    return img

def test():
    img = cv2.imread("images/image8.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = 50, 60
    vertical = True
    dark = False
    brightness = 100
    line_amplitude =40
    mask_width =10
    frequency = 0.5
    gamma = 0.4
    variance = 1
    noise = False
    blur = True
    image = blur_lines(gray, (x, y), vertical, dark, brightness, line_amplitude, mask_width, frequency, gamma, variance, noise)
    #image = add_noise(gray, prob=0.9)


    image = cv2.resize(image, (960, 960)) 
    height, width = image.shape

    lambd = frequency*mask_width
    w = mask_width/2
    n = int(w/lambd)+1
    X_centers = []
    print(n)
    Xc = x+mask_width/2
    Yc = y + line_amplitude/2
    if variance<0.8:
        mask_width = variance*mask_width*1.1
    else:
        mask_width = variance*mask_width
    if gamma-variance>0.2:
        line_amplitude = variance*line_amplitude/gamma
    if n<=5 and mask_width>10:
        for i in range(n+1):
            xcp = lambd*i-lambd/4
            print(np.exp(-((xcp**2)*0.5/(variance*w*2)**2)))
            print("xp")
            if np.exp(-(xcp**2)*0.5/(variance*w*2)**2) >= 0.45 and xcp<w:
                X_centers.append(xcp+lambd/2)

            xcm = -lambd*i+lambd/4
            print(np.exp(-(xcm**2)*0.5/((variance*w)**2)))
            print("xm")
            if np.exp(-(xcm**2)*0.5/(variance*w*2)**2) >= 0.45 and -xcm<w:
                X_centers.append(xcm)
        print(mask_width/2, lambd/4)


        for xc in X_centers:
            cv2.rectangle(image, (int((Xc+xc-lambd/4)*width/100), int((Yc-line_amplitude/2)*height/100)), (int((Xc+xc+lambd/4)*width/100), int((Yc+line_amplitude/2)*height/100)), color=(255,255,255), thickness=2)

    else:
        cv2.rectangle(image, (int((Xc-mask_width/2)*width/100), int((Yc-line_amplitude/2)*height/100)), (int((Xc+mask_width/2)*width/100), int((Yc+line_amplitude/2)*height/100)), color=(255,255,255), thickness=2)



    cv2.imshow("image", image)
    #cv2.imshow("mask", mask)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


test()
    