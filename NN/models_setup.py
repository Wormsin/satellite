import utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch

def resnet101(train_dir, test_dir, batch_size, lr):
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

    train_loader, classes = utils.data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = utils.data_setup(test_dir, test_transform, batch_size)
    resnet_model = models.resnet101(weights='DEFAULT')
    for param in resnet_model.parameters():
        param.requires_grad = False
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, len(classes)),
        #nn.Sigmoid()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.fc.parameters(), lr)
    resnet_model = resnet_model.to(device)

    return resnet_model, optimizer, loss_fn, device, train_loader, test_loader, classes

def efficientnet(train_dir, test_dir, batch_size, lr):
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(480, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(480, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_loader, classes = utils.data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = utils.data_setup(test_dir, test_transform, batch_size)
    effic_model = models.efficientnet_v2_l(weights='DEFAULT')
    for param in effic_model.parameters():
        param.requires_grad = False
    effic_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True), 
        nn.Linear(in_features=effic_model.classifier[1].in_features, 
                        out_features=len(classes), 
                        bias=True),)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(effic_model.classifier.parameters(), lr)
    effic_model = effic_model.to(device)

    return effic_model, optimizer, loss_fn, device, train_loader, test_loader, classes

def swin_vit(train_dir, test_dir, batch_size, lr):
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
    train_loader, classes = utils.data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = utils.data_setup(test_dir, test_transform, batch_size)
    swin_model = models.swin_v2_s(weights='DEFAULT')
    for param in swin_model.parameters():
        param.requires_grad = False
    swin_model.head = nn.Sequential( 
        nn.Linear(in_features=swin_model.head.in_features, 
                        out_features=len(classes), 
                        bias=True),)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(swin_model.head.parameters(), lr)
    swin_model = swin_model.to(device)

    return swin_model, optimizer, loss_fn, device, train_loader, test_loader, classes

def vit(train_dir, test_dir, batch_size, lr):
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
    train_loader, classes = utils.data_setup(train_dir, train_transform, batch_size)
    test_loader, _ = utils.data_setup(test_dir, test_transform, batch_size)
    vit_model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
    for param in vit_model.parameters():
        param.requires_grad = False
    vit_model.heads = nn.Sequential( 
        nn.Linear(in_features=vit_model.hidden_dim, 
                        out_features=len(classes), 
                        bias=True),)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.heads.parameters(), lr)
    vit_model = vit_model.to(device)

    return vit_model, optimizer, loss_fn, device, train_loader, test_loader, classes

