import torch
from PIL import Image
import numpy as np
import os 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def checkpoint_fn(model, optimizer, filename):
    torch.save({'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),}, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def accuracy_fn(y_pred, y_true):
    n_correct = torch.eq(y_pred, y_true).sum().item()
    accuracy = (n_correct / len(y_pred))
    return accuracy

def get_metrics(y_true, y_pred, classes):
    """Per-class performance metrics."""
    # Performance
    performance = {"overall": {}, "class": {}}
    # Overall performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance["overall"]["precision"] = metrics[0]
    performance["overall"]["recall"] = metrics[1]
    performance["overall"]["f1"] = metrics[2]
    performance["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    return performance

def binary_metrics_test(y_true, y_pred, loss):
    acc = accuracy_score(y_true, y_pred)
    if acc>0.95 and loss<0.25:
        return []
    else: 
        return [0]

def multi_metrics_test(y_true, y_pred, classes):
    f1 = f1_score(y_true, y_pred, average=None)
    if np.sum(f1>0.85) == len(classes):
        return []
    else:
        bad_mask = (f1<0.85).nonzero()
        bad_results = [classes[int(i)] for i in bad_mask[0] ]
        return bad_results

def metrics_test(model_name, y_true, y_pred, classes, loss):
    match model_name:
        case 'resnet101':
            return binary_metrics_test(y_true, y_pred, loss)
        case 'swin_vit':
            return multi_metrics_test(y_true, y_pred, classes)

def train(num_epochs, optimizer, model, loss_fn, train_loader, test_loader, device, checkpoint, name, classes):
    Image.MAX_IMAGE_PIXELS = 124010496
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    for epoch in np.arange(0, num_epochs):
        if epoch%50 ==0 and epoch!=0 and checkpoint:
            checkpoint_fn(model, optimizer, f'weights/{name}_{epoch}.pth')
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
            print(f"Epoch: {epoch} | train loss: {train_loss:.2f}")
        result = metrics_test(name, y_true, y_pred, classes, train_loss)
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

def eval_metrics(loader, model, classes):
    Image.MAX_IMAGE_PIXELS = 124010496
    y_pred = [] # save predction
    y_true = [] # save ground truth
    # iterate over data
    for inputs, labels in loader:
        #inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)  
        output = output.argmax(dim=1).float()
        y_pred.extend(output)  
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) 
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    performance = get_metrics(y_true=y_true, y_pred=y_pred, classes=classes)
    df_overall = pd.DataFrame(performance['overall'], index = [0])
    df_classes =pd.DataFrame(performance['class'])
    return sn.heatmap(df_cm, annot=True).get_figure(), df_overall, df_classes

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

