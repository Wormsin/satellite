import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, models

def data_setup(dir, transform, batch_size):
    # Load datasets
    dataset = datasets.ImageFolder(root=dir, transform=transform)
    # Create data loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    classes = dataset.classes
    return loader, classes

def get_transform(model_name):
    match model_name:
        case 'binary':
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
        case 'multi':
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
    return train_transform, test_transform

def get_model_optim(model_name, classes, lr = 0.001):
    ncls = len(classes)
    match model_name:
        case 'binary':
            model = models.resnet101(weights = 'DEFAULT')
            model.fc = nn.Sequential( nn.Linear(model.fc.in_features, ncls),)
            optimizer = optim.Adam(model.fc.parameters(), lr)
        case 'multi':
            model = models.swin_v2_s(weights='DEFAULT')
            model.head = nn.Sequential( nn.Linear(model.head.in_features, ncls),)
            optimizer = optim.Adam(model.head.parameters(), lr)
    return model, optimizer

def load_weights(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def model4train(name,train_dir,test_dir, batch_size, lr):
    train_transform, test_transform = get_transform(model_name=name)
    train_loader, classes = data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = data_setup(test_dir, test_transform, batch_size)
    model, optimizer = get_model_optim(name, classes, lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, optimizer, loss_fn, device, train_loader, test_loader, classes

def model4classify(name, classes, weights):
    _, transform = get_transform(model_name=name)
    model, _ = get_model_optim(name, classes)
    model = load_weights(model, weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model , transform, device
