import utils
import setup
from torchvision import transforms

train_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/dataset_classes/train'
test_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/dataset_classes/test'
#model_path="weights/classification_model.pth"
batch_size = 32
lr = 0.001
epochs = 300

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

model, optimizer, loss_fn, device = setup.load_model(classes=classes, lr=lr)

utils.train(num_epochs=epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, 
            test_loader=test_loader, device=device, checkpoint=False)


