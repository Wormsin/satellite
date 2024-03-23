import utils
import models_setup
import import_models
import matplotlib.pyplot as plt

train_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/dataset_classes/train'
test_dir = '/home/msvermis/Downloads/ML_projects/satellite-project/Data/dataset_classes/test'
batch_size = 32
lr = 0.001
epochs = 300

model, optimizer, loss_fn, device, train_loader, test_loader, classes = models_setup.resnet_setup(train_dir, test_dir, batch_size, lr)
#model, test_loader, transform, classes = import_models.resnet101(model_path="weights/classification_model.pth", test_dir=test_dir, batch_size=batch_size, checkpoint=False)

print(f"classes: {classes}")
print(f"device: {device}")

utils.train(num_epochs=epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, 
            test_loader=test_loader, device=device, checkpoint=True)

#_, df_overall, df_classes  = utils.evaluate(loader=test_loader, model=model, classes=classes)
#print(df_overall.to_markdown())
#print(df_classes.to_markdown())
#plt.title(f'ResNet101, batch_size = {batch_size}, lr = {lr}, epochs = {epochs}')
#plt.show()

#image_path = 'images/im7_def.jpg'
#utils.test(image_path, transform, model, classes)