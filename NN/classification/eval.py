import utils
import models
import matplotlib.pyplot as plt
import params

args = params.parameters()

test_dir =  args.dataset +'/test'
weights = args.weights
name = args.name
batch_size = args.batch
lr = args.lr
epochs = args.epochs

model, test_loader, classes = models.model4eval(name=name, weights = weights, test_dir=test_dir, 
                                        device='cpu', batch_size=batch_size)

print(f"classes: {classes}")

_, df_overall, df_classes  = utils.eval_metrics(loader=test_loader, model=model, classes=classes)
print(df_overall.to_markdown())
print(df_classes.to_markdown())
plt.title(f'vit_b_16, batch_size = {batch_size}, lr = {lr}, epochs = {epochs}')
plt.show()