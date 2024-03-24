import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim


def load_model(lr, classes):
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

    return resnet_model, optimizer, loss_fn, device

def import_model(model_path, classes, checkpoint):
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
    return model




