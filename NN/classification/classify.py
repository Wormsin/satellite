import utils
import models as m
import params
import os

if __name__ == "__main__":
    args = params.parameters()


    dir = args.dataset
    model1 = args.weights[0]
    model2 =  args.weights[1]
    classes2 = args.classes
    device = args.device

    classes1 = ['defected', 'clear']
    name1 = args.name[0]
    name2 = args.name[1]

    model1, transform1 = m.model4classify(name=name1, classes=classes1, weights=model1, device=device)
    model2, transform2 = m.model4classify(name=name2, classes=classes2, weights=model2, device=device)


    utils.classification(dir = dir, transform = transform1, model= model1, classes= classes1, device=device)
    utils.classification(dir = classes1[0], transform = transform2, model= model2, classes= classes2, device=device)

    os.rmdir(classes1[0])
    os.rmdir(classes1[1])
