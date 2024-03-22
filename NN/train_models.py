import utils
import models_setup

train_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/resnet_dataset/train'
test_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/resnet_dataset/test'
batch_size = 128
lr = 0.001
epochs = 50

model, optimizer, loss_fn, device, train_loader, test_loader, classes = models_setup.resnet_setup(train_dir, test_dir, batch_size, lr)

utils.train(num_epochs=epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, 
            test_loader=test_loader, device=device, checkpoint=True)
