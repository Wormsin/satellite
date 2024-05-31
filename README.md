# Data processing
## decode L15 files
Folder with L15 files: satellite/L15\
Result folder with images: satellite/images
```ruby
go run decode_L15/seek.go 
```
## binary classification 
Example of weights path: satellite/weights/resnet101.pth<br>
Default folder for the new images: satellite/images<br>
Default folder for the defected images: satellite/defected
```ruby
python -m classification.classify --name resnet101
```
## multiclass classification
```ruby
python -m classification.classify --name swin_vit --classes ВПВ ВПП ОИ ЧПИ
```

## Train the model
Dataset for trainning: satellite/resnet101_dataset
```ruby
python -m classification.train --name resnet101 --epochs 500
```
***Cuda is always used if available***.

## names of defects
ВПП -- вертикальные полосы по скану<br>
ОИ -- отсутствие изображения<br>
ЧПИ -- частичное потеря информации 





