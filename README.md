# Data processing
## configure the directory
Сreating the necessary directories:<br>
L15 files: **L15**<br>
New images: **images**<br>
New defected images: **new_defected**<br>
Datasets for binary and multiclass classification with classes from classes.txt: **datasets**
```ruby
python -m settings.create_storage
```
## decode L15 files
```ruby
go run decode_L15/seek.go 
```
## binary classification 
After classification the images are added to the corresponding dataset and *new_defected*
```ruby
python -m classification.classify --name resnet101
```
## multiclass classification
After classification the images are added to the corresponding dataset from *new_defected*<br>
Classes are taken from the classification/classes.txt
```ruby
python -m classification.classify --name swin_vit 
```
## Add the new class
The new class will be used after training before this images in the new class are added manually without classification
```ruby
./classification/add_new_class.sh NEW_CLASS
```
## Train the model
```ruby
python -m classification.train --name resnet101 --epochs 500
```
***Cuda is always used if available***.

## names of defects
ВПМ -- вертикальные полосы множественные<br>
ВПП -- вертикальные полосы по скану<br>
ОИ -- отсутствие изображения<br>
ЧПИ -- частичное потеря информации 





