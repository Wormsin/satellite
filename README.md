<p align="center">
  <img src ="https://github.com/Wormsin/satellite/assets/142012648/09d811e8-bc4a-4ead-b724-5befb0e0a185"  width="188" height="187">
  <img src = "https://github.com/Wormsin/satellite/assets/142012648/edc7d002-2ade-45fa-baf4-940ea054f73e" width="188" height="187">
  <img src ="https://github.com/Wormsin/satellite/assets/142012648/64dc4f92-c85b-4951-85fb-8e6107796a25"  width="188" height="187">
</p>

# Processing satellite images
## Extracting an image from a L15 file
```ruby
go run decode_L15/seek.go --save DIR_FOR_IMAGES --source DIR_WITH_L15_FILES
```
## Generating defects
To increase and diversify the dataset, defects were generated in accordance with the documentation.
Line defects were performed using a Gabor filter.
```ruby
python defects/main.py
```
Images are generated with bounding boxes for yolov5 object detection. 
## 4-class classification
**Yolov5** performs poorly because our dataset is too small and contains many identical images. Therefore, we decided to use a classification approach for this task. \
We first do binary classification to separate defective and non-defective images using **resnet101**, and then 4-class classification with the option of adding a new class. \
Here are the metrics of three pretrained models for 4-class classification:
### resnet101

<p align="center">
  <img src="https://github.com/Wormsin/satellite/assets/142012648/708d55ca-f8eb-4660-b9f3-39e9bfbb5ef5" width="469" height="309">
  <img src= "https://github.com/Wormsin/satellite/assets/142012648/0c0e143d-1e32-49e1-99d8-56182ef2f58d" width="248" height="91">
</p>

### swin-vit

<p align="center">
  <img src = "https://github.com/Wormsin/satellite/assets/142012648/1693d39e-f92d-4f66-8d00-b7eea401e28d" width="462" height="304">
  <img src= "https://github.com/Wormsin/satellite/assets/142012648/605ce8db-f37f-4751-b7e5-5f0888bd8dd3" width="239" height="93">
</p>

### vit

<p align="center">
  <img src="https://github.com/Wormsin/satellite/assets/142012648/36a99cb4-7a43-45b7-99e8-4867bd693127" width="464" height="304">
  <img src= "https://github.com/Wormsin/satellite/assets/142012648/6974752c-f4d4-4353-8efa-949ab3cf3866" width="255" height="89">
</p>

##### _classes_: ВПМ -- _little lines_, ВПП -- _big lines_, ОИ -- _lack of information_, ЧПИ -- _partial lack of information_





