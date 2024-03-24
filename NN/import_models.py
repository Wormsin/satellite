import torch
import torch.nn as nn
from torchvision import transforms, models
import utils 


def resnet101(model_path, test_dir, batch_size, checkpoint):
    transform = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_loader, classes = utils.data_setup(test_dir, transform, batch_size)
    model = models.resnet101(weights = 'DEFAULT')  # assuming you saved the entire model including architecture
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, len(classes)),
    )
    if checkpoint:
        checkpoint_w = torch.load(model_path)
        model.load_state_dict(checkpoint_w['model'])
    else:
        model.load_state_dict(torch.load(model_path))
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    model.eval()
    return model, test_loader, transform, classes

def swin_vit(model_path, test_dir, batch_size, checkpoint):
    transform = transforms.Compose([
        transforms.Resize(260, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_loader, classes = utils.data_setup(test_dir, transform, batch_size)
    model = models.swin_v2_s(weights='DEFAULT')
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, len(classes)),
    )
    if checkpoint:
        checkpoint_w = torch.load(model_path)
        model.load_state_dict(checkpoint_w['model'])
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, test_loader, transform, classes


