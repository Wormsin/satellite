
## Classify the data
```ruby
python classify.py --dataset DATASET_PATH --weights MODELS_WEIGHTS_PATH --classes ВПВ ВПП ОИ ЧПИ
                    --name 'resnet101' 'swin_vit' --device cuda or cpu 
```

## Train the model
```ruby
python train.py --dataset DATASET_PATH --name 'resnet101' --device cuda or cpu --batch BATCH_SIZE --lr LEARNING_RATE --epochs EPOCHS
```