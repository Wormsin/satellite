import utils
import models
import params

args = params.parameters()

train_dir = args.dataset + '/train'
test_dir =  args.dataset +'/test'
name = args.name[0]
batch_size = args.batch
lr = args.lr
epochs = args.epochs

model, optimizer, loss_fn, device, train_loader, test_loader, classes = models.model4train(name=name, train_dir=train_dir, test_dir=test_dir, 
                                                                                           batch_size=batch_size, lr=lr)

print(f"classes: {classes}")
print(f"device: {device}")

utils.train(num_epochs=epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, 
            test_loader=test_loader, device=device, checkpoint=True)
