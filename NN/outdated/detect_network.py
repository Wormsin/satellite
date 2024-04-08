import torch
import cv2
from PIL import Image
import os
import shutil


#model_path = './weights2201.pt'
model_path = '/home/msvermis/Downloads/ML_projects/Yolo/yolov5/satellite/last.pt'
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path,  )
model = torch.hub.load('/home/msvermis/Downloads/ML_projects/Yolo/yolov5', 'custom', path=model_path, source='local') 


path_folder = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/big_imgs'  # Replace with the path to your image
model.conf = 0.2
save_def_folder = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/defected'
save_undef_folder = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/undefected'


def detect_defect(img, name, old_root):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, size = 1392)
    has_annotations = len(results.xyxy[0])
    
    if has_annotations>0:
        result_folder = save_def_folder
    else:
        result_folder =  save_undef_folder

    #annotated_image = results.render()[0]  
    save_path = os.path.join(result_folder, f'{name[:-4]}.jpg')
    old_path = os.path.join(old_root, f'{name[:-4]}.jpg')
    #cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    shutil.move(old_path, save_path)


if not os.path.exists(save_def_folder):
    os.mkdir(save_def_folder)
    os.mkdir(save_undef_folder)
for root, dirs, files, rootfd in os.fwalk(path_folder):
    if len(files)!=0:
        print(f'folder number: {root}')
        for i in files:
            all_path = root + "/"+ i
            if i[-2:]!="db" and i[-3:]!="zip":
                img = cv2.imread(all_path)
                detect_defect(img, i, root)

