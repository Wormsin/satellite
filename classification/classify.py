import classification.utils as utils
import classification.models as m
import classification.params as params
import os


def add2dataset(classes, name):
    for cl in classes:
        if cl == 'defected':
            if not os.path.isdir('new_defected'):
                os.mkdir('new_defected')
            os.system(f'cp -r {cl}/* new_defected')
        images =  os.listdir(cl)
        if images != []:
            split_index = int(0.8 * len(images))
            train = images[:split_index]
            test = images[split_index:]
            for img in train:
                os.rename(os.path.join(cl, img), os.path.join(f'datasets/{name}/train/{cl}', img))
            if test != []:
                for img in test:
                    os.rename(os.path.join(cl, img), os.path.join(f'datasets/{name}/test/{cl}', img))
        os.rmdir(f'{cl}')


if __name__ == "__main__":
    args = params.parameters()

    with open('classification/classes.txt', 'r') as file:
        content = file.read()
        classes = content.split()

    names = args.type
    classes, dir = (classes, 'new_defected') if names[0] != 'binary' else (['defected', 'normal'], 'images')

    if not len(os.listdir(dir)) == 0:
        model, transform, device = m.model4classify(name=names[0], classes=classes, weights=f'weights/{names[0]}'+'.pth')
        utils.classification(dir = dir, transform = transform, model= model, classes= classes, device=device)
        add2dataset(classes, names[0]) 
    else:
        print(f'{dir} is empty!')

