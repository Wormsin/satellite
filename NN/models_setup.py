import utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch

def resnet_setup(train_dir, test_dir, batch_size, lr):
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

    train_loader, test_loader, classes = utils.data_setup(train_dir, test_dir, train_transform, test_transform, batch_size)
    resnet_model = models.resnet101(weights='DEFAULT')
    # Freeze the layers of the pre-trained model
    for param in resnet_model.parameters():
        param.requires_grad = False
    # Modify the classifier part of the model for binary classification
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, len(classes)),
        nn.Sigmoid()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.fc.parameters(), lr)
    resnet_model = resnet_model.to(device)

    return resnet_model, optimizer, loss_fn, device, train_loader, test_loader, classes