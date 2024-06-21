import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

Image.MAX_IMAGE_PIXELS = 124010496

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
            train_transform = v2.Compose([
    v2.RandomResizedCrop(224, interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomPerspective(distortion_scale=0.6, p=0.5),
    v2.RandomAdjustSharpness(sharpness_factor=2),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            test_transform = v2.Compose([
        v2.Resize(232, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(224),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        case 'multi':
            train_transform = v2.Compose([
    v2.RandomResizedCrop(256, interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomPerspective(distortion_scale=0.6, p=0.5),
    v2.RandomAdjustSharpness(sharpness_factor=2),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            test_transform = v2.Compose([
        v2.Resize(260, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(256),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
