import torch
from PIL import Image
import numpy as np
import os 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def binary_metrics_test(y_true, y_pred, loss, epoch):
    acc = accuracy_score(y_true, y_pred)
    print(f"Epoch: {epoch} | train loss: {loss:.2f}, accuracy: {acc:.2f}")
    if acc>0.95 and loss<0.15:
        return []
    else: 
        return [0]

def multi_metrics_test(y_true, y_pred, classes, epoch):
    f1 = f1_score(y_true, y_pred, average=None)
    print(f"Epoch: {epoch} | F1 for {classes}: {f1}")
    if np.sum(f1>0.85) == len(classes):
        return []
    else:
        bad_mask = (f1<0.85).nonzero()
        bad_results = [classes[int(i)] for i in bad_mask[0] ]
        return bad_results

def metrics_test(model_name, y_true, y_pred, classes, loss, epoch):
    match model_name:
        case 'binary':
            return binary_metrics_test(y_true, y_pred, loss, epoch)
        case 'multi':
            return multi_metrics_test(y_true, y_pred, classes, epoch)

def train(num_epochs, optimizer, model, loss_fn, train_loader, test_loader, device, name, classes):
    Image.MAX_IMAGE_PIXELS = 124010496
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    for epoch in np.arange(0, num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss/=len(train_loader)

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1).data.cpu().numpy()
                y_pred.extend(predicted)
                labels = labels.data.cpu().numpy()
                y_true.extend(labels)
        if epoch%1==0:
            result = metrics_test(name, y_true, y_pred, classes, train_loss, epoch)
        if result == []:
            print(f"Metrics is good, save the new weights!")
            torch.save(model.state_dict(), f'weights/{name}.pth')
            return 1
    if result[0] == 0:
        print(f"Metrics are bad, so the weights stay the same")
        return 0
    else:
        print(f"More images need to be added to classes {result}, so the weights without the new classes remain")
        return 0

def test(image_path, transform, model, classes, device):
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension
    # Make predictions
    input_image = input_image.to(device)
    with torch.no_grad():
        output = model(input_image)
    # Interpret the output
    prediction = output.argmax(dim=1).int()
    return classes[prediction]

def classification(dir, transform, model, classes, device):
    Image.MAX_IMAGE_PIXELS = 124010496
    images = os.listdir(dir)
    for cl in classes:
        if not os.path.isdir(cl):
            os.mkdir(cl)
    for image in images:
        img_path = dir+'/'+image
        prediction = test(image_path=img_path, transform=transform, 
             model = model, classes=classes, device=device)
        os.rename(img_path, os.path.join(prediction, image))
    print(f'Successful classification into {classes} classes')


def check_batch_size_memory(model, batch_size, input_shape, device='cuda'):
    try:
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        model.to(device)  
        _ = model(dummy_input)
        print(f"This batch size {batch_size} fits into GPU memory.")
        return False
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory error: the batch size {batch_size} is too large for the GPU.")
        return True
    finally:
        torch.cuda.empty_cache()