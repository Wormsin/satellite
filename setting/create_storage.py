import os 

def create(folder):
    folder = '../' + folder
    if not os.path.isdir(folder):
        os.makedirs(folder)


folders = ['datasets']
branches2 = ['resnet101', 'swin_vit']
branches3 = ['test', 'train']
branches4 = [['defected', 'normal'], ['ВПМ', 'ВПП', 'ОИ', 'ЧПИ']]

create('images')
create('L15')
create('weights')
create('new_defected')

for f in folders:
    for i, b2 in enumerate(branches2):
        for b3 in branches3:
            for b4 in branches4[i]:
                dir = '/'.join([f, b2, b3, b4])
                create(dir)
