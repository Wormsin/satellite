import torch
from PIL import Image
import numpy as np
import os 
import matplotlib as plt 
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def data_setup(train_dir, test_dir, train_transform, test_transform, batch_size):
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    classes = train_dataset.classes
    return train_loader, test_loader, classes

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

def train(num_epochs, optimizer, model, loss_fn, train_loader, test_loader, device, checkpoint):
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    writer = SummaryWriter(log_dir="/home/msvermis/Downloads/ML_projects/satellite-project/satellite/NN/logs")
    if not os.path.isdir('weights') and checkpoint:
        os.mkdir('weights')
    for epoch in np.arange(0, num_epochs):
        if epoch%50 ==0 and epoch!=0 and checkpoint:
            checkpoint_fn(model, optimizer, f'weights/classification_model_{epoch}.pth')
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred.squeeze(), labels.float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss/=len(train_loader)

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            correct = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                correct += accuracy_fn(predicted, labels)
            accuracy = correct / len(test_loader)
        if epoch%1==0:
            print(f"Epoch: {epoch} | train loss: {train_loss:.2f}, test accuracy: {accuracy:.1f}")
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"test_acc": accuracy}, global_step=epoch)
        # Close the writer
        writer.close()
    torch.save(model.state_dict(), 'weights/classification_model.pth')

def evaluate(loader, model, classes):
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
    performance = get_metrics(y_true=labels, y_pred=y_pred, classes=classes)
    return sn.heatmap(df_cm, annot=True).get_figure(), performance

def test(image_path, transform, model):
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension
    # Make predictions
    with torch.no_grad():
        output = model(input_image)
    # Interpret the output
    prediction = 'with defects' if output.item() < 0.5 else 'without defects'
    print(f'The image {image_path} is predicted to be: {prediction}')


