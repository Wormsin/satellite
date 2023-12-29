import cv2
import numpy as np

def get_line_mask(brightness:int, line_amplitude:int, mask_width: int, num: int):
    mask = np.zeros((mask_width, line_amplitude))

    line_space = mask_width//num
    line_width = line_space//2
    for i in range(0, mask.shape[0], line_space):
        mask[i:i+line_width]+=1
    
    return mask


def add_lines(img: np.uint, x: int, y:int, vertical: bool, dark:bool, brightness: int, line_amplitude:int, mask_width: float, num: int):
    image0 = np.copy(img)
    height, width = img.shape
    line_amplitude = height*line_amplitude//100
    mask_width = int(width*mask_width//100)
    x = x*width//100
    y = y*height//100
    mask = get_line_mask(brightness, line_amplitude, mask_width, num)

    if vertical:
        mask = mask.T
        line_x, line_y = mask_width, line_amplitude
    else:
        line_y, line_x = mask_width, line_amplitude
    
    bw_mask = ((mask*255-img[y:y+line_y, x:x+line_x])*brightness//100)*mask
    bw_mask = bw_mask.astype(np.uint8)
    
    img[y:y+line_y, x:x+line_x]+=bw_mask
    img[image0==0] = 0
    return img


img = cv2.imread("images/image4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = add_lines(img=gray, x=50, y=0, vertical=True, dark=False, brightness=20, line_amplitude=100, mask_width=4.5, num=4)

image = cv2.resize(image, (960, 960)) 
cv2.imshow("mask", image)
cv2.waitKey(0) 
cv2.destroyAllWindows()