import torch
import torch.nn as nn
from torchvision import transforms, models
import utils 
import torch.optim as optim

def get_transform(model_name):
    match model_name:
        case 'resnet101':
            train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            test_transform = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        case 'swin_vit':
            train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            test_transform = transforms.Compose([
        transforms.Resize(260, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        case 'vit':
            train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            test_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform

def get_model_optim(model_name, classes, lr = 0.001, checkpoint = False, device_check = True):
    ncls = len(classes) if len(classes)>2 else 1
    match model_name:
        case 'resnet101':
            model = models.resnet101(weights = 'DEFAULT')
            model.fc = nn.Sequential( nn.Linear(model.fc.in_features, ncls),)
            optimizer = optim.Adam(model.fc.parameters(), lr)
        case 'swin_vit':
            model = models.swin_v2_s(weights='DEFAULT')
            model.head = nn.Sequential( nn.Linear(model.head.in_features, ncls),)
            optimizer = optim.Adam(model.head.parameters(), lr)
        case 'vit':
            model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            model.heads = nn.Sequential( nn.Linear(in_features=model.hidden_dim, 
                                                   out_features=ncls, bias=True))
            optimizer = optim.Adam(model.heads.parameters(), lr)
    return model, optimizer

def load_weights(model, model_path, checkpoint = False, device_check = False):
    if checkpoint:
        checkpoint_w = torch.load(model_path)
        model.load_state_dict(checkpoint_w['model'])
    else:
        model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() and device_check else 'cpu')
    model = model.to(device)
    model.eval()
    return model

def model4train(name,train_dir,test_dir, batch_size, lr):
    train_transform, test_transform = get_transform(model_name=name)
    train_loader, classes = utils.data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = utils.data_setup(test_dir, test_transform, batch_size)
    model, optimizer = get_model_optim(name, classes, lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, optimizer, loss_fn, device, train_loader, test_loader, classes

def model4classify(name, classes, weights, device):
    _, transform = get_transform(model_name=name)
    model, _ = get_model_optim(name, classes)
    model = load_weights(model, weights, device_check=device == 'cuda')
    return model , transform

def model4eval(name, weights, device, test_dir, batch_size, checkpoint):
    _, transform = get_transform(model_name=name)
    test_loader, classes = utils.data_setup(test_dir, transform, batch_size)
    model, _ = get_model_optim(name, classes)
    model = load_weights(model, weights, checkpoint=checkpoint, device_check=device == 'cuda')
    return model, test_loader, classes
