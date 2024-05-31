import classification.utils as utils
import classification.models as m
import classification.params as params
import os

if __name__ == "__main__":
    args = params.parameters()


    #device = args.device

    classes, dir = (args.classes, 'defected') if args.classes != [] else (['defected', 'clear'], 'images')
    names = args.name

    if not len(os.listdir(dir)) == 0:
        model, transform, device = m.model4classify(name=names[0], classes=classes, weights=f'weights/{names[0]}'+'.pth')
        utils.classification(dir = dir, transform = transform, model= model, classes= classes, device=device) 
    else:
        print(f'{dir} is empty!')


    #os.rmdir('defected')
