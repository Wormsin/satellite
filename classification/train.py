import classification.utils as utils
import classification.models as models
import classification.params as params

if __name__ == "__main__":
    args = params.parameters()
    
    name = args.type
    train_dir = f'datasets/{name}' + '/train'
    test_dir =  f'datasets/{name}' +'/test'
    batch_size = args.batch
    lr = args.lr
    epochs = args.epochs
    mem_error = True

    while mem_error:
        model, optimizer, loss_fn, device, train_loader, test_loader, classes = models.model4train(name=name, train_dir=train_dir, test_dir=test_dir, 
                                                                                                batch_size=batch_size, lr=lr)
        for images, _ in train_loader:
            input_shape = images.size()
            break
        input_shape = tuple(input_shape[1:])
        mem_error = utils.check_batch_size_memory(model=model, batch_size=batch_size, input_shape=input_shape)
        batch_size=batch_size//2
 
    print(f"classes: {classes}")
    print(f"device: {device}")
    

    result = utils.train(num_epochs=epochs, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, 
                test_loader=test_loader, device=device, name=name, classes=classes)

    if name == 'multi' and result:
        with open('classification/classes.txt', 'w') as file:
            file.write(" ".join(classes))
